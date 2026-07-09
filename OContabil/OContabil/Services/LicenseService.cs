using System.IO;
using System.Security.Cryptography;
using System.Text.Json;

namespace OContabil.Services;

/// <summary>
/// Licença de avaliação offline com expiração. O arquivo license.ocl carrega um
/// payload assinado (ECDSA P-256) pela chave privada do dono do produto — que
/// NUNCA acompanha o app. Aqui só vive a chave pública: sem assinatura válida ou
/// com prazo vencido, o login é bloqueado (gate no WebBridge, lado C#, fora do
/// alcance da UI). Proteção básica anti-retrocesso de relógio via marca d'água
/// DPAPI do último instante visto. Escopo honesto: barreira de avaliação, não DRM
/// — quem descompilar e recompilar o binário remove qualquer gate client-side.
/// </summary>
public static class LicenseService
{
    // Par gerado pelo OContabil.LicenseTool (keygen). Rotacionar a chave privada
    // exige atualizar este bloco e recompilar — licenças antigas deixam de valer.
    private const string PublicKeyPem = @"-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEMAoqwVA+XH6uI/LITN9iCtEJCwV1
4W56tLK1wk6crwf7MQIr8kE9znbn9FGq0hlIWopI8i9Dh8QwlD57VI3cyg==
-----END PUBLIC KEY-----";

    private static readonly string Root = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OContabil");

    private static string ClockPath => Path.Combine(Root, "secrets", "lic_clock.bin");

    public sealed record Status(bool Valida, string? Motivo, string? Nome, DateTime? ExpiraUtc)
    {
        public int DiasRestantes => ExpiraUtc.HasValue
            ? Math.Max(0, (int)Math.Ceiling((ExpiraUtc.Value - DateTime.UtcNow).TotalDays))
            : 0;
    }

    public static Status Check()
    {
        try
        {
            var path = FindLicenseFile();
            if (path == null)
                return new Status(false,
                    "Licença não encontrada. Solicite um arquivo license.ocl e coloque em " +
                    @"%LocalAppData%\OContabil\ (ou na pasta do OContabil.exe).", null, null);

            using var envelope = JsonDocument.Parse(File.ReadAllText(path));
            var payloadB64 = envelope.RootElement.GetProperty("payload").GetString() ?? "";
            var sigB64 = envelope.RootElement.GetProperty("signature").GetString() ?? "";
            var payload = Convert.FromBase64String(payloadB64);

            using var ecdsa = ECDsa.Create();
            ecdsa.ImportFromPem(PublicKeyPem);
            if (!ecdsa.VerifyData(payload, Convert.FromBase64String(sigB64), HashAlgorithmName.SHA256))
                return new Status(false, "Licença inválida (assinatura não confere).", null, null);

            using var doc = JsonDocument.Parse(payload);
            var p = doc.RootElement;
            var nome = p.TryGetProperty("nome", out var n) ? n.GetString() : null;
            if (!p.TryGetProperty("expiraUtc", out var e) || !e.TryGetDateTime(out var expira))
                return new Status(false, "Licença inválida (sem prazo de expiração).", nome, null);

            var agora = DateTime.UtcNow;
            if (ClockRolledBack(agora))
                return new Status(false,
                    "Relógio do sistema retrocedido em relação ao último uso — corrija a data/hora.", nome, expira);
            if (agora > expira)
                return new Status(false,
                    $"Licença de avaliação expirada em {expira.ToLocalTime():dd/MM/yyyy}. " +
                    "Solicite uma nova para continuar.", nome, expira);

            return new Status(true, null, nome, expira);
        }
        catch (Exception ex)
        {
            SafeLog.Error("license.check", ex);
            return new Status(false, "Licença ilegível ou corrompida.", null, null);
        }
    }

    private static string? FindLicenseFile()
    {
        foreach (var candidate in new[]
        {
            Path.Combine(Root, "license.ocl"),
            Path.Combine(AppContext.BaseDirectory, "license.ocl"),
        })
            if (File.Exists(candidate)) return candidate;
        return null;
    }

    // Marca d'água monotônica: guarda (cifrado por DPAPI) o maior UtcNow já visto.
    // Voltar o relógio mais de 24h aquém dela invalida a sessão — barra o truque
    // trivial de "volto a data e a licença nunca vence". Tolerância de 24h evita
    // falso positivo com fuso/horário de verão/ajuste legítimo.
    private static bool ClockRolledBack(DateTime agoraUtc)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ClockPath)!);
            long stored = 0;
            if (File.Exists(ClockPath))
            {
                var plain = Crypto.Unprotect(File.ReadAllText(ClockPath));
                long.TryParse(plain, out stored);
            }

            if (stored > 0 && agoraUtc.Ticks < stored - TimeSpan.TicksPerDay)
                return true;

            if (agoraUtc.Ticks > stored)
                File.WriteAllText(ClockPath, Crypto.Protect(agoraUtc.Ticks.ToString()));
            return false;
        }
        catch (Exception ex)
        {
            SafeLog.Error("license.clock", ex);
            return false; // falha do mecanismo auxiliar não pode trancar usuário legítimo
        }
    }
}
