#!/usr/bin/env python
"""Your own server for the smell-report map. Standard library only.

    set KTM_ADMIN_PASSWORD=choose-a-long-passphrase
    python report_server.py serve  --static out --db reports.db --port 8000
    python report_server.py export --db reports.db --out observations_web.csv     # for `calibrate --obs`
    python report_server.py backup --db reports.db --out backups/reports-2026-11-20.db
    python report_server.py purge  --db reports.db --before 2026-01-01
    python report_server.py stats  --db reports.db

Who can do what
  Everyone (no login)   view the map, add a report (POST /api/report), see report counts (GET /api/stats).
                        They cannot read, edit or delete any report.
  You (admin login)     http://HOST:PORT/admin  : list and filter reports, edit, hide spam, delete, summary,
                        export CSV. Needs KTM_ADMIN_PASSWORD (or --admin-password). With no password the
                        whole admin side is switched off.

Security notes
  * Admin sign-in sets an HttpOnly, SameSite=Strict cookie (8 hours). Changes must be JSON sent from the
    same site; wrong-site requests are refused. 5 wrong passwords in 15 minutes locks that address out.
  * Report text comes from strangers. The admin page shows it as plain text, never as HTML.
  * IP addresses are held in memory for rate limiting only, never written down.
  * The database must live outside --static or it would be downloadable; the server refuses to start otherwise.
  * Plain HTTP only. Put it behind a TLS-terminating proxy (Caddy, nginx, a platform with https), pass
    --trust-proxy and --secure-cookies. Without https, the admin password and cookie travel in clear text:
    only use the admin side over https or on localhost / a trusted network. Browsers also only allow
    "my location" on https pages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

NPT = timezone(timedelta(hours=5, minutes=45))          # Nepal time: what the page records
COLS = ["lat", "lon", "datetime", "detected", "intensity_0_5", "wind_dir_from_deg", "session", "observer",
        "river", "map_says_inside", "exposure_m"]          # observations.csv columns for `calibrate`, plus extras
EDITABLE = ["lat", "lon", "datetime", "detected", "intensity_0_5", "river", "map_says_inside", "exposure_m"]
S, N, W, E = 27.3, 28.0, 85.0, 85.7                      # reports outside this box are rejected
MAX_BODY, MAX_ADMIN_BODY = 2048, 32768
SESSION_S = 8 * 3600
SCHEMA = """CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY, received_utc TEXT NOT NULL,
  lat REAL NOT NULL, lon REAL NOT NULL, datetime TEXT NOT NULL,
  detected INTEGER NOT NULL CHECK(detected IN (0,1)), intensity_0_5 INTEGER, wind_dir_from_deg REAL,
  session TEXT, observer TEXT, river TEXT, map_says_inside INTEGER, exposure_m INTEGER,
  status TEXT NOT NULL DEFAULT 'ok', edited_utc TEXT,
  UNIQUE(lat, lon, datetime, detected))"""
DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
HERE = Path(__file__).resolve().parent


def clean(r: object, now: datetime | None, recent: bool = True) -> dict:
    """Validate and normalise one report. Raises ValueError with a short reason.
    recent=False skips the 'happened in the last day' rule (used when you edit an old report)."""
    if not isinstance(r, dict):
        raise ValueError("body must be a JSON object")

    def num(k, lo, hi):
        v = r.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"{k} missing or out of range")
        return float(v)

    lat, lon = round(num("lat", S, N), 4), round(num("lon", W, E), 4)
    det = r.get("detected")
    if det not in (0, 1) or isinstance(det, bool):
        raise ValueError("detected must be 0 or 1")
    dt = r.get("datetime")
    if not isinstance(dt, str) or not DT_RE.match(dt):
        raise ValueError("datetime must look like 2026-07-01 22:30")
    try:
        when = datetime.strptime(dt, "%Y-%m-%d %H:%M").replace(tzinfo=NPT)
    except ValueError:
        raise ValueError("datetime is not a real time")
    if recent and not now - timedelta(hours=24) <= when <= now + timedelta(minutes=15):
        raise ValueError("datetime is not recent")
    inten = r.get("intensity_0_5")
    if inten in (None, ""):
        inten = 0 if det == 0 else None
    elif isinstance(inten, bool) or inten not in (0, 1, 2, 3, 4, 5):
        raise ValueError("intensity_0_5 must be 0-5")
    elif det == 0:
        inten = 0
    inside = r.get("map_says_inside")
    inside = inside if inside in (0, 1) and not isinstance(inside, bool) else None
    ex = r.get("exposure_m")
    ex = int(ex) if isinstance(ex, (int, float)) and not isinstance(ex, bool) and 0 <= ex <= 5000 else None
    river = r.get("river") if isinstance(r.get("river"), str) else ""
    river = re.sub(r"[\x00-\x1f\x7f]", "", river).strip()[:80]
    return {"lat": lat, "lon": lon, "datetime": dt, "detected": int(det), "intensity_0_5": inten,
            "wind_dir_from_deg": None, "session": dt[:10] + "-web", "observer": "web", "river": river,
            "map_says_inside": inside, "exposure_m": ex}


class Store:
    def __init__(self, path: Path):
        self.path = path
        with self._con() as con:
            con.execute(SCHEMA)
            have = {r[1] for r in con.execute("PRAGMA table_info(reports)")}
            if "status" not in have:                       # databases made by the first version
                con.execute("ALTER TABLE reports ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'")
            if "edited_utc" not in have:
                con.execute("ALTER TABLE reports ADD COLUMN edited_utc TEXT")
            con.execute("CREATE TABLE IF NOT EXISTS admin_log(id INTEGER PRIMARY KEY, at_utc TEXT NOT NULL, "
                        "action TEXT NOT NULL, report_id INTEGER, detail TEXT)")

    def _con(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log(self, con, action, rid=None, detail=""):
        con.execute("INSERT INTO admin_log(at_utc, action, report_id, detail) VALUES(?,?,?,?)",
                    (self._now(), action, rid, detail[:500]))

    # ---- public side
    def add(self, rec: dict) -> bool:
        with self._con() as con:
            cur = con.execute("INSERT OR IGNORE INTO reports(received_utc," + ",".join(COLS) + ") VALUES(?,"
                              + ",".join("?" * len(COLS)) + ")", [self._now()] + [rec[c] for c in COLS])
            return cur.rowcount == 1

    def public_stats(self) -> dict:
        with self._con() as con:
            n, k = con.execute("SELECT COUNT(*), COALESCE(SUM(detected),0) FROM reports WHERE status='ok'").fetchone()
        return {"reports": n, "smelled": int(k)}

    # ---- admin side
    def csv(self, include_hidden=False) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(COLS)
        with self._con() as con:
            q = "SELECT " + ",".join(COLS) + " FROM reports" + ("" if include_hidden else " WHERE status='ok'") + " ORDER BY id"
            for row in con.execute(q):
                w.writerow(["" if v is None else v for v in tuple(row)])
        return buf.getvalue()

    def list(self, status="ok", smell="", river="", q="", limit=50, offset=0):
        where, args = [], []
        if status in ("ok", "hidden"):
            where.append("status=?"); args.append(status)
        if smell in ("0", "1"):
            where.append("detected=?"); args.append(int(smell))
        if river:
            where.append("COALESCE(NULLIF(river,''),'(none)')=?"); args.append(river)
        if q:
            where.append("(river LIKE ? OR datetime LIKE ? OR CAST(id AS TEXT)=?)"); args += [f"%{q}%", f"%{q}%", q]
        w = (" WHERE " + " AND ".join(where)) if where else ""
        with self._con() as con:
            total = con.execute("SELECT COUNT(*) FROM reports" + w, args).fetchone()[0]
            rows = [dict(r) for r in con.execute(
                "SELECT id, received_utc, edited_utc, status, " + ",".join(EDITABLE) + " FROM reports" + w +
                " ORDER BY id DESC LIMIT ? OFFSET ?", args + [limit, offset])]
        return {"total": total, "rows": rows}

    def get(self, rid: int):
        with self._con() as con:
            r = con.execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
        return dict(r) if r else None

    def update(self, rid: int, rec: dict) -> bool:
        with self._con() as con:
            sets = ",".join(f"{c}=?" for c in COLS)
            cur = con.execute(f"UPDATE reports SET {sets}, edited_utc=? WHERE id=?", [rec[c] for c in COLS] + [self._now(), rid])
            if cur.rowcount:
                self.log(con, "edit", rid, json.dumps({k: rec[k] for k in EDITABLE}))
            return cur.rowcount == 1

    def bulk(self, ids: list[int], action: str) -> int:
        if not ids:
            return 0
        ph = ",".join("?" * len(ids))
        with self._con() as con:
            if action == "delete":
                cur = con.execute(f"DELETE FROM reports WHERE id IN ({ph})", ids)
            else:
                cur = con.execute(f"UPDATE reports SET status=? WHERE id IN ({ph})", ["hidden" if action == "hide" else "ok"] + ids)
            self.log(con, action, None, f"{cur.rowcount} report(s): {ids[:50]}")
            return cur.rowcount

    def summary(self) -> dict:
        with self._con() as con:
            st = {r[0]: r[1] for r in con.execute("SELECT status, COUNT(*) FROM reports GROUP BY status")}
            n, k, m, agree = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(detected),0), SUM(map_says_inside IS NOT NULL), "
                "COALESCE(SUM(map_says_inside IS NOT NULL AND map_says_inside=detected),0) FROM reports WHERE status='ok'").fetchone()
            rivers = [dict(r) for r in con.execute(
                "SELECT COALESCE(NULLIF(river,''),'(none)') AS river, COUNT(*) AS n, SUM(detected) AS smelled, "
                "SUM(map_says_inside IS NOT NULL) AS compared, SUM(map_says_inside IS NOT NULL AND map_says_inside=detected) AS agree "
                "FROM reports WHERE status='ok' GROUP BY 1 ORDER BY n DESC")]
            days = [dict(r) for r in con.execute(
                "SELECT substr(datetime,1,10) AS day, COUNT(*) AS n, SUM(detected) AS smelled FROM reports WHERE status='ok' "
                "GROUP BY 1 ORDER BY 1 DESC LIMIT 14")]
            log = [dict(r) for r in con.execute("SELECT at_utc, action, report_id, detail FROM admin_log ORDER BY id DESC LIMIT 15")]
        return {"status": st, "visible": n, "smelled": int(k), "compared": int(m or 0), "agree": int(agree),
                "rivers": rivers, "days": days, "log": log}

    def purge_before(self, day: str) -> int:
        with self._con() as con:
            cur = con.execute("DELETE FROM reports WHERE substr(datetime,1,10) < ?", (day,))
            self.log(con, "purge", None, f"before {day}: {cur.rowcount}")
            return cur.rowcount

    def backup(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(out)
        with self._con() as src:
            src.backup(dst)
        dst.close()


class RateLimit:
    """Sliding window per key, in memory only."""
    def __init__(self, n: int, window_s: float):
        self.n, self.window, self.hits, self.lock = n, window_s, defaultdict(deque), threading.Lock()

    def _trim(self, q, now):
        while q and now - q[0] > self.window:
            q.popleft()

    def take(self, who: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[who]
            self._trim(q, now)
            if len(q) >= self.n:
                return False
            q.append(now)
            if len(self.hits) > 5000:
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window]:
                    del self.hits[k]
            return True

    def blocked(self, who: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[who]
            self._trim(q, now)
            return len(q) >= self.n

    def add(self, who: str):
        with self.lock:
            self.hits[who].append(time.monotonic())


class Handler(SimpleHTTPRequestHandler):
    server_version = "ktm-reports/2"

    def log_message(self, fmt, *args):                    # no addresses, no query strings
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

    # ---- helpers
    def _headers(self, ctype, length, extra=None):
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)

    def _send(self, code, body: bytes, ctype="application/json", extra=None, cors=False):
        self.send_response(code)
        self._headers(ctype, len(body), extra)
        if cors and self.server.cors:
            self.send_header("Access-Control-Allow-Origin", self.server.cors)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj, **kw):
        self._send(code, json.dumps(obj).encode(), **kw)

    def _who(self):
        if self.server.trust_proxy:
            xf = self.headers.get("X-Forwarded-For", "")
            if xf:
                return xf.split(",")[0].strip()
        return self.client_address[0]

    def _read_json(self, limit):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if not 0 < n <= limit:
            raise ValueError("body too large or empty")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ---- admin auth
    def _sign(self, exp: int) -> str:
        return hmac.new(self.server.secret, str(exp).encode(), hashlib.sha256).hexdigest()

    def _is_admin(self) -> bool:
        if not self.server.password:
            return False
        auth = self.headers.get("Authorization", "")                  # for scripts: Authorization: Bearer <password>
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), self.server.password.encode()):
            return True
        for part in self.headers.get("Cookie", "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "ktm_admin":
                exp, _, sig = v.partition(".")
                if exp.isdigit() and int(exp) > time.time() and sig not in self.server.revoked \
                        and hmac.compare_digest(sig, self._sign(int(exp))):
                    return True
        return False

    def _revoke_current(self):
        for part in self.headers.get("Cookie", "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "ktm_admin":
                self.server.revoked.add(v.partition(".")[2])

    def _same_site_json(self) -> bool:
        """State-changing admin calls: JSON content type (cannot be sent by a cross-site form) and same Origin."""
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return False
        org = self.headers.get("Origin")
        return not org or urlparse(org).netloc == self.headers.get("Host", "")

    def _need_admin(self, write=False) -> bool:
        if not self.server.password:
            self._json(404, {"ok": False, "error": "not found"})
            return False
        if not self._is_admin():
            self._json(401, {"ok": False, "error": "sign in required"})
            return False
        if write and not self._same_site_json():
            self._json(403, {"ok": False, "error": "refused: requests must be JSON from this site"})
            return False
        return True

    # ---- routes
    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        if self.server.cors:                               # only the public report endpoint is ever cross-site
            self.send_header("Access-Control-Allow-Origin", self.server.cors)
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/report":
            return self._post_report()
        if path == "/api/admin/login":
            return self._login()
        if path == "/api/admin/logout":
            if self._need_admin(write=True):
                self._revoke_current()                       # a copied cookie stops working too, not just this browser's
                return self._send(200, b'{"ok": true}', extra={"Set-Cookie": self._cookie("", 0)})
            return
        if path == "/api/admin/reports/bulk":
            if not self._need_admin(write=True):
                return
            try:
                b = self._read_json(MAX_ADMIN_BODY)
                ids = [int(i) for i in b.get("ids", [])][:1000]
                act = b.get("action")
                if act not in ("hide", "unhide", "delete"):
                    raise ValueError("action must be hide, unhide or delete")
            except (ValueError, TypeError, AttributeError) as e:
                return self._json(400, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": True, "changed": self.server.store.bulk(ids, act)})
        return self._json(404, {"ok": False, "error": "not found"})

    def _cookie(self, value, max_age):
        c = f"ktm_admin={value}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}"
        return c + ("; Secure" if self.server.secure_cookies else "")

    def _login(self):
        if not self.server.password:
            return self._json(404, {"ok": False, "error": "not found"})
        if not self._same_site_json():
            return self._json(403, {"ok": False, "error": "refused: requests must be JSON from this site"})
        who = self._who()
        if self.server.login_limit.blocked(who):
            return self._json(429, {"ok": False, "error": "too many wrong passwords; wait 15 minutes"})
        try:
            pw = self._read_json(1024).get("password", "")
        except (ValueError, AttributeError):
            return self._json(400, {"ok": False, "error": "bad request"})
        if not isinstance(pw, str) or not hmac.compare_digest(pw.encode(), self.server.password.encode()):
            self.server.login_limit.add(who)
            time.sleep(0.4)
            return self._json(401, {"ok": False, "error": "wrong password"})
        exp = int(time.time()) + SESSION_S
        self._send(200, b'{"ok": true}', extra={"Set-Cookie": self._cookie(f"{exp}.{self._sign(exp)}", SESSION_S)})

    def _post_report(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            return self._json(413, {"ok": False, "error": "body too large or empty"}, cors=True)
        if not self.server.limit.take(self._who()):
            return self._json(429, {"ok": False, "error": "too many reports from this address; try again later"}, cors=True)
        try:
            rec = clean(json.loads(self.rfile.read(n).decode("utf-8")), datetime.now(NPT))
        except (ValueError, UnicodeDecodeError) as e:
            return self._json(400, {"ok": False, "error": str(e)}, cors=True)
        stored = self.server.store.add(rec)
        self._json(201 if stored else 200, {"ok": True, "duplicate": not stored}, cors=True)

    def do_PATCH(self):
        m = re.fullmatch(r"/api/admin/reports/(\d+)", self.path.split("?")[0])
        if not m:
            return self._json(404, {"ok": False, "error": "not found"})
        if not self._need_admin(write=True):
            return
        rid, st = int(m.group(1)), self.server.store
        cur = st.get(rid)
        if not cur:
            return self._json(404, {"ok": False, "error": "no such report"})
        try:
            patch = self._read_json(MAX_ADMIN_BODY)
            if not isinstance(patch, dict) or set(patch) - set(EDITABLE):
                raise ValueError("editable fields: " + ", ".join(EDITABLE))
            rec = clean({**{k: cur[k] for k in EDITABLE}, **patch}, None, recent=False)
            if not st.update(rid, rec):
                raise ValueError("no such report")
        except sqlite3.IntegrityError:
            return self._json(409, {"ok": False, "error": "another report already has the same place, time and answer"})
        except (ValueError, TypeError) as e:
            return self._json(400, {"ok": False, "error": str(e)})
        self._json(200, {"ok": True, "report": {k: rec[k] for k in EDITABLE}})

    def do_DELETE(self):
        m = re.fullmatch(r"/api/admin/reports/(\d+)", self.path.split("?")[0])
        if not m:
            return self._json(404, {"ok": False, "error": "not found"})
        if not self._need_admin(write=True):
            return
        self._json(200, {"ok": True, "changed": self.server.store.bulk([int(m.group(1))], "delete")})

    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        one = lambda k, d="": (qs.get(k) or [d])[0]
        if path == "/api/health":
            return self._json(200, {"ok": True})
        if path == "/api/stats":
            return self._json(200, self.server.store.public_stats(), cors=True)
        if path == "/admin":
            if not self.server.password:
                return self._json(404, {"ok": False, "error": "not found"})
            nonce = secrets.token_urlsafe(16)
            html = self.server.admin_html.replace("__NONCE__", nonce).encode("utf-8")
            return self._send(200, html, "text/html; charset=utf-8", {
                "Content-Security-Policy": f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                                           "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                "X-Frame-Options": "DENY"})
        if path == "/api/export":
            if not self._need_admin():
                return
            body = self.server.store.csv(include_hidden=one("all") == "1").encode("utf-8")
            return self._send(200, body, "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="observations_web.csv"'})
        if path == "/api/admin/reports":
            if not self._need_admin():
                return
            try:
                lim = max(1, min(200, int(one("limit", "50")))); off = max(0, int(one("offset", "0")))
            except ValueError:
                return self._json(400, {"ok": False, "error": "limit and offset must be numbers"})
            return self._json(200, self.server.store.list(one("status", "ok"), one("smell"), one("river"), one("q")[:80], lim, off))
        if path == "/api/admin/summary":
            if not self._need_admin():
                return
            return self._json(200, self.server.store.summary())
        if path.startswith("/api/"):
            return self._json(404, {"ok": False, "error": "not found"})
        return super().do_GET()


def serve(a) -> int:
    static, db = Path(a.static).resolve(), Path(a.db).resolve()
    if not static.is_dir():
        print(f"--static {static} is not a folder (run the map script first)", file=sys.stderr)
        return 2
    if static in db.parents or db == static:
        print("The database must live outside the static folder or it would be downloadable.", file=sys.stderr)
        return 2
    db.parent.mkdir(parents=True, exist_ok=True)
    pw = a.admin_password or os.environ.get("KTM_ADMIN_PASSWORD", "")
    if pw and len(pw) < 10:
        print("Admin password is too short: use at least 10 characters.", file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(static)))
    srv.store, srv.limit = Store(db), RateLimit(a.rate, 3600)
    srv.login_limit, srv.revoked = RateLimit(5, 900), set()
    srv.password, srv.cors, srv.trust_proxy, srv.secure_cookies = pw, a.cors_origin, a.trust_proxy, a.secure_cookies
    srv.secret = os.environ.get("KTM_SECRET", "").encode() or secrets.token_bytes(32)   # sessions end on restart unless KTM_SECRET is set
    srv.admin_html = (HERE / "web" / "admin.html").read_text(encoding="utf-8") if pw else ""
    print(f"Map and public report endpoint: http://{a.host}:{a.port}/   (database {db})")
    print(f"Admin: http://{a.host}:{a.port}/admin" if pw else "No KTM_ADMIN_PASSWORD set: the admin side is OFF (use the export/backup/purge commands).")
    if pw and a.host not in ("127.0.0.1", "localhost") and not a.secure_cookies:
        print("WARNING: admin over plain http on a network address. Use https (a proxy) and --secure-cookies.", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--static", default="out", help="folder with odour_map.html (default: out)")
    s.add_argument("--db", default="reports.db")
    s.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to listen on all interfaces")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--rate", type=int, default=20, help="reports per address per hour")
    s.add_argument("--admin-password", help="or set KTM_ADMIN_PASSWORD (at least 10 characters)")
    s.add_argument("--cors-origin", default="", help="let another site post reports here, e.g. https://you.github.io")
    s.add_argument("--trust-proxy", action="store_true", help="read the visitor address from X-Forwarded-For (behind a proxy)")
    s.add_argument("--secure-cookies", action="store_true", help="mark the admin cookie Secure (use when served over https)")
    e = sub.add_parser("export"); e.add_argument("--db", default="reports.db"); e.add_argument("--out", default="observations_web.csv")
    e.add_argument("--all", action="store_true", help="include hidden reports")
    b = sub.add_parser("backup"); b.add_argument("--db", default="reports.db"); b.add_argument("--out", required=True)
    p = sub.add_parser("purge"); p.add_argument("--db", default="reports.db"); p.add_argument("--before", required=True, help="YYYY-MM-DD: delete reports dated earlier")
    t = sub.add_parser("stats"); t.add_argument("--db", default="reports.db")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return serve(a)
    if not Path(a.db).exists():
        print(f"{a.db} does not exist yet", file=sys.stderr)
        return 2
    st = Store(Path(a.db))
    if a.cmd == "stats":
        print(json.dumps(st.summary(), indent=2))
    elif a.cmd == "backup":
        st.backup(Path(a.out)); print(f"backup written to {a.out}")
    elif a.cmd == "purge":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.before):
            print("--before must look like 2026-01-01", file=sys.stderr); return 2
        print(f"deleted {st.purge_before(a.before)} reports dated before {a.before}")
    else:
        Path(a.out).write_text(st.csv(include_hidden=a.all), encoding="utf-8", newline="")
        print(f"wrote {a.out} ({st.public_stats()['reports']} visible reports{', hidden included' if a.all else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
