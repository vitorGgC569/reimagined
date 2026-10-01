using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Web.WebView2.Core;
using OContabil.Data;
using OContabil.Services;

namespace OContabil.Views;

// WebView2 host for the OContabil web design (WebUI/).  Maps the local WebUI
// folder to a virtual host (a file:// origin blocks the babel/JSX fetches),
// navigates to it, and bridges the design to the real C# services via
// postMessage <-> WebBridge.
public partial class WebShellWindow : Window
{
    private readonly AuthService _auth = new();
    private readonly WebBridge _bridge;

    public WebShellWindow()
    {
        InitializeComponent();
        using (var db = new AppDbContext()) DbInitializer.Initialize(db);
        _bridge = new WebBridge(_auth);
        Loaded += async (_, _) => await InitAsync();
    }

    private async Task InitAsync()
    {
        try
        {
            await web.EnsureCoreWebView2Async();

            // ── Hardening do WebView2 (defesa em profundidade, menor superfície) ──
            var s = web.CoreWebView2.Settings;
            s.AreDevToolsEnabled = false;               // sem DevTools/inspeção em produção
            s.AreDefaultContextMenusEnabled = false;    // sem menu "Inspecionar/Salvar como"
            s.AreBrowserAcceleratorKeysEnabled = false; // sem F12/Ctrl+Shift+I/Ctrl+P etc.
            s.IsZoomControlEnabled = false;
            s.IsStatusBarEnabled = false;
            s.IsGeneralAutofillEnabled = false;
            s.IsPasswordAutosaveEnabled = false;

            // Apenas a origem local é permitida — bloqueia navegação externa, popups e downloads.
            web.CoreWebView2.NavigationStarting += (_, e) =>
            {
                if (!e.Uri.StartsWith("https://ocontabil.local/", StringComparison.OrdinalIgnoreCase))
                {
                    e.Cancel = true;
                    SafeLog.Warn("webview.nav", "Navegacao externa bloqueada: " + e.Uri);
                }
            };
            web.CoreWebView2.NewWindowRequested += (_, e) => { e.Handled = true; };
            web.CoreWebView2.DownloadStarting += (_, e) => { e.Cancel = true; };

            web.CoreWebView2.WebMessageReceived += OnWebMessage;

            var webUi = Path.Combine(AppContext.BaseDirectory, "WebUI");
            web.CoreWebView2.SetVirtualHostNameToFolderMapping(
                "ocontabil.local", webUi, CoreWebView2HostResourceAccessKind.Allow);
            web.CoreWebView2.Navigate("https://ocontabil.local/index.html");
        }
        catch (Exception ex)
        {
            SafeLog.Error("webview.init", ex);
            MessageBox.Show("Não foi possível iniciar a interface. Tente reabrir o aplicativo.", "OContabil",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void OnWebMessage(object sender, CoreWebView2WebMessageReceivedEventArgs e)
    {
        string id = null;
        try
        {
            var root = JsonDocument.Parse(e.WebMessageAsJson).RootElement;
            if (root.TryGetProperty("id", out var idEl)) id = idEl.GetString();
            var action = root.GetProperty("action").GetString();
            var payload = root.TryGetProperty("payload", out var pl) ? pl : default;
            Respond(id, _bridge.Dispatch(action, payload));
        }
        catch (Exception ex)
        {
            SafeLog.Error("webmsg", ex);
            Respond(id, new { ok = false, error = "Não foi possível processar a solicitação." });
        }
    }

    private void Respond(string id, object result)
    {
        try { web.CoreWebView2.PostWebMessageAsJson(JsonSerializer.Serialize(new { id, result })); }
        catch { /* janela fechando */ }
    }
}
