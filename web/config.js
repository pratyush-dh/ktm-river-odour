/* Where smell reports are sent. Empty = reports stay in each visitor's browser (they can export a CSV).

   A) Map hosted on GitHub Pages (static, cannot receive reports itself): deploy report_endpoint.gs as a Google
      Apps Script web app and use
          window.KTM_CONFIG = { report_url: "https://script.google.com/macros/s/XXXX/exec", report_mode: "no-cors" };
   B) Map served by report_server.py (any host that can run Python): reports go to the same site
          window.KTM_CONFIG = { report_url: "/api/report", report_mode: "cors" };
*/
window.KTM_CONFIG = { report_url: "", report_mode: "cors" };
