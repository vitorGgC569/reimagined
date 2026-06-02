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
            web.CoreWebView2.WebMessageReceived += OnWebMessage;

            var webUi = Path.Combine(AppContext.BaseDirectory, "WebUI");
            web.CoreWebView2.SetVirtualHostNameToFolderMapping(
                "ocontabil.local", webUi, CoreWebView2HostResourceAccessKind.Allow);
            web.CoreWebView2.Navigate("https://ocontabil.local/index.html");
        }
        catch (Exception ex)
        {
            MessageBox.Show("Falha ao iniciar a UI web: " + ex.Message, "OContabil",
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
            Respond(id, new { ok = false, error = ex.Message });
        }
    }

    private void Respond(string id, object result)
    {
        try { web.CoreWebView2.PostWebMessageAsJson(JsonSerializer.Serialize(new { id, result })); }
        catch { /* janela fechando */ }
    }
}
