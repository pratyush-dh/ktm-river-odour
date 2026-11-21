/**
 * Receives smell reports from the map and appends them to a Google Sheet. Works with a map hosted on
 * GitHub Pages (static hosting cannot receive reports itself, so the reports go here instead).
 * Columns match observations.csv for `kathmandu_odour_map.py calibrate`, plus received_utc.
 *
 * Setup (a few minutes, needs your Google account; written and logic-tested here, not deployed from here):
 *   1. Create a Google Sheet. Extensions > Apps Script. Paste this whole file over Code.gs and save.
 *   2. Deploy > New deployment > type "Web app". Execute as: Me. Who has access: Anyone.
 *      Authorise when asked. Copy the web app URL (ends in /exec).
 *   3. In the site's config.js set:  report_url: "<that URL>",  report_mode: "no-cors"
 *      (the browser cannot read this script's reply, so the page shows "Sent" once it is handed over).
 *   4. File > Download > CSV from the sheet, then:  calibrate --obs that.csv --by-river
 *   After you edit this script you must Deploy > Manage deployments > Edit > New version for it to take effect.
 *
 * Privacy: the map sends only a spot rounded to ~10 m, a time, the nearest river's name and a yes/no.
 * The sheet is private to you; keep it that way. Reports are anonymous (Apps Script never shows senders' addresses).
 *
 * Abuse: anyone with the URL can post. This screens each report (inside the valley box, recent time, valid
 * values), drops exact duplicates, and stops accepting after MAX_PER_HOUR reports in an hour in total (it cannot
 * rate-limit per person). Look at the data before fitting a model.
 */
const COLS = ['lat', 'lon', 'datetime', 'detected', 'intensity_0_5', 'wind_dir_from_deg',
              'session', 'observer', 'river', 'map_says_inside', 'exposure_m'];
const BOX = { S: 27.3, N: 28.0, W: 85.0, E: 85.7 };
const MAX_PER_HOUR = 300;

function clean_(r, nowMs) {
  if (!r || typeof r !== 'object' || Array.isArray(r)) throw new Error('body must be an object');
  const num = (k, lo, hi) => {
    const v = r[k];
    if (typeof v !== 'number' || !isFinite(v) || v < lo || v > hi) throw new Error(k + ' missing or out of range');
    return v;
  };
  const lat = Math.round(num('lat', BOX.S, BOX.N) * 1e4) / 1e4;
  const lon = Math.round(num('lon', BOX.W, BOX.E) * 1e4) / 1e4;
  if (r.detected !== 0 && r.detected !== 1) throw new Error('detected must be 0 or 1');
  const dt = r.datetime;
  if (typeof dt !== 'string' || !/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/.test(dt)) throw new Error('datetime format');
  const when = Date.parse(dt.replace(' ', 'T') + ':00+05:45');          // the page records Kathmandu time
  if (isNaN(when)) throw new Error('datetime is not a real time');
  if (when < nowMs - 24 * 3600e3 || when > nowMs + 15 * 60e3) throw new Error('datetime is not recent');
  let inten = r.intensity_0_5;
  if (inten === null || inten === undefined || inten === '') inten = r.detected === 0 ? 0 : '';
  else if ([0, 1, 2, 3, 4, 5].indexOf(inten) < 0) throw new Error('intensity_0_5 must be 0-5');
  else if (r.detected === 0) inten = 0;
  const inside = (r.map_says_inside === 0 || r.map_says_inside === 1) ? r.map_says_inside : '';
  const ex = (typeof r.exposure_m === 'number' && r.exposure_m >= 0 && r.exposure_m <= 5000) ? Math.round(r.exposure_m) : '';
  const river = (typeof r.river === 'string' ? r.river : '').replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 80);
  return { lat: lat, lon: lon, datetime: dt, detected: r.detected, intensity_0_5: inten, wind_dir_from_deg: '',
           session: dt.slice(0, 10) + '-web', observer: 'web', river: river, map_says_inside: inside, exposure_m: ex };
}

function reply_(text) { return ContentService.createTextOutput(text); }

function doPost(e) {
  let rec;
  try {
    if (!e || !e.postData || !e.postData.contents || e.postData.contents.length > 2048) throw new Error('bad body');
    rec = clean_(JSON.parse(e.postData.contents), Date.now());
  } catch (err) { return reply_('rejected: ' + err.message); }
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const cache = CacheService.getScriptCache();
    const hourKey = 'n' + Utilities.formatDate(new Date(), 'UTC', 'yyyyMMddHH');
    const n = Number(cache.get(hourKey) || 0);
    if (n >= MAX_PER_HOUR) return reply_('busy: try again later');
    const dupKey = 'd' + [rec.lat, rec.lon, rec.datetime, rec.detected].join('|');
    if (cache.get(dupKey)) return reply_('duplicate');
    const sh = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sh.getLastRow() === 0) sh.appendRow(COLS.concat(['received_utc']));
    sh.appendRow(COLS.map(function (c) { return rec[c]; }).concat([new Date().toISOString()]));
    cache.put(hourKey, String(n + 1), 3700);
    cache.put(dupKey, '1', 21600);
    return reply_('ok');
  } finally { lock.releaseLock(); }
}

/** Open the web app URL in a browser to see a count (no report contents are shown). */
function doGet() {
  const sh = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  return reply_(JSON.stringify({ reports: Math.max(0, sh.getLastRow() - 1) }));
}
