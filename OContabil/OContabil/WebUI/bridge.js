/* ============================================================
   OContabil — ponte com os serviços C# (via WebView2 postMessage).
   window.OContabilBridge.call(action, payload) -> Promise<result>.
   result: { ok: true, data } | { ok: false, error }.
   Fora do WebView2 (ex.: aberto no navegador), `available` = false e as
   chamadas resolvem com erro — as telas caem no mock de data.js.
   ============================================================ */
(function () {
  var seq = 0;
  var pending = {};
  var hasWV = !!(window.chrome && window.chrome.webview);

  if (hasWV) {
    window.chrome.webview.addEventListener('message', function (e) {
      var msg = e.data;
      if (msg && msg.id != null && pending[msg.id]) {
        var resolve = pending[msg.id];
        delete pending[msg.id];
        resolve(msg.result);
      }
    });
  }

  window.OContabilBridge = {
    available: hasWV,
    call: function (action, payload) {
      return new Promise(function (resolve) {
        if (!hasWV) { resolve({ ok: false, error: 'bridge indisponível (fora do app)' }); return; }
        var id = String(++seq);
        pending[id] = resolve;
        window.chrome.webview.postMessage({ id: id, action: action, payload: payload || {} });
        // timeout defensivo
        setTimeout(function () {
          if (pending[id]) { delete pending[id]; resolve({ ok: false, error: 'timeout' }); }
        }, 15000);
      });
    }
  };
})();
