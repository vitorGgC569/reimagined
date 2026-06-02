using System;
using System.Linq;
using System.Text.Json;
using OContabil.Data;

namespace OContabil.Services;

// Bridge between the embedded web design (WebView2) and the real C# services.
// The web UI posts {id, action, payload}; WebShellWindow routes the action here
// and posts the JSON result back.  Each action maps to an existing service so
// the design becomes functional without duplicating business logic.
//
// Result convention: { ok: true, data: <...> } or { ok: false, error: "msg" }.
public sealed class WebBridge
{
    private readonly AuthService _auth;

    public WebBridge(AuthService auth) => _auth = auth;

    public object Dispatch(string action, JsonElement payload)
    {
        switch (action)
        {
            case "login": return Login(payload);
            case "session": return Session();
            case "logout": _auth.Logout(); return Ok(new { });

            case "clients.list": return ClientsList();
            case "documents.list": return DocumentsList(payload);

            default: return Err("Ação desconhecida: " + action);
        }
    }

    // ── Auth ──
    private object Login(JsonElement p)
    {
        var user = Str(p, "user");
        var pass = Str(p, "pass");
        if (string.IsNullOrWhiteSpace(user) || string.IsNullOrEmpty(pass))
            return Err("Preencha usuário e senha.");

        var outcome = _auth.Login(user, pass);
        if (outcome.IsSuccess && outcome.User != null)
            return Ok(UserDto(outcome.User));

        if (outcome.Status == LoginStatus.Throttled)
        {
            var min = Math.Max(1, (int)Math.Ceiling(outcome.BlockedFor.TotalMinutes));
            return Err($"Excesso de tentativas. Tente novamente em {min} min.");
        }
        var msg = outcome.RemainingAttempts is > 0 and < 5
            ? $"Usuário ou senha incorretos. ({outcome.RemainingAttempts} tentativa(s) restante(s))"
            : "Usuário ou senha incorretos.";
        return Err(msg);
    }

    private object Session() =>
        _auth.IsLoggedIn && _auth.CurrentUser != null ? Ok(UserDto(_auth.CurrentUser)) : Err("sem sessão");

    private static object UserDto(Models.User u) => new
    {
        id = u.Id,
        nome = u.FullName,
        usuario = u.Username,
        papel = u.Role.ToString(),
        canManageUsers = u.Role == Models.UserRole.Admin,
    };

    // ── Read endpoints (clientes/documentos) ──
    private object ClientsList()
    {
        using var db = new AppDbContext();
        var rows = db.Clients.OrderBy(c => c.Name).Select(c => new
        {
            id = c.Id,
            razaoSocial = c.Name,
            cnpj = c.Cnpj,
            regime = c.TaxRegime,
            ativo = c.IsActive,
            volume = c.DocumentCount,
        }).ToList();
        return Ok(rows);
    }

    private object DocumentsList(JsonElement p)
    {
        using var db = new AppDbContext();
        var rows = db.Documents
            .OrderByDescending(d => d.Id)
            .Select(d => new
            {
                id = d.Id,
                arquivo = d.Filename,
                tipo = d.DocumentType,
                clienteId = d.ClientId,
                status = d.StatusDisplay,
                confianca = d.ConfidenceScore,
            })
            .Take(500)
            .ToList();
        return Ok(rows);
    }

    // ── helpers ──
    private static object Ok(object data) => new { ok = true, data };
    private static object Err(string error) => new { ok = false, error };

    private static string Str(JsonElement p, string key) =>
        p.ValueKind == JsonValueKind.Object && p.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() ?? "" : "";
}
