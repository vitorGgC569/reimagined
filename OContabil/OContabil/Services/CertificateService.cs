using System.IO;
using System.Security.Cryptography.X509Certificates;

namespace OContabil.Services;

/// <summary>
/// Manages A1 Digital Certificates (.pfx) for SEFAZ integration.
/// </summary>
public class CertificateService
{
    private readonly string _storePath;
    public X509Certificate2? CurrentCertificate { get; private set; }

    public CertificateService()
    {
        _storePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "OContabil", "certificates");
        Directory.CreateDirectory(_storePath);
    }

    public bool HasCertificate => CurrentCertificate != null && CurrentCertificate.NotAfter > DateTime.Now;

    public string CertificateInfo
    {
        get
        {
            if (CurrentCertificate == null) return "Nenhum certificado carregado";
            return $"{CurrentCertificate.Subject}\n" +
                   $"Valido ate: {CurrentCertificate.NotAfter:dd/MM/yyyy}\n" +
                   $"Emissor: {CurrentCertificate.Issuer}";
        }
    }

    /// <summary>
    /// Load a .pfx certificate with password.
    /// </summary>
    public CertificateLoadResult LoadCertificate(string pfxPath, string password)
    {
        try
        {
            if (!File.Exists(pfxPath))
                return CertificateLoadResult.Error("Arquivo nao encontrado.");

            var ext = Path.GetExtension(pfxPath).ToLowerInvariant();
            if (ext != ".pfx" && ext != ".p12")
                return CertificateLoadResult.Error("Formato invalido. Use arquivo .pfx ou .p12");

            var cert = new X509Certificate2(pfxPath, password,
                X509KeyStorageFlags.MachineKeySet | X509KeyStorageFlags.PersistKeySet);

            if (cert.NotAfter < DateTime.Now)
                return CertificateLoadResult.Error(
                    $"Certificado vencido em {cert.NotAfter:dd/MM/yyyy}. Renove com a Autoridade Certificadora.");

            if (!cert.HasPrivateKey)
                return CertificateLoadResult.Error("Certificado sem chave privada. Necessario certificado A1 completo.");

            CurrentCertificate = cert;

            return new CertificateLoadResult
            {
                Success = true,
                Subject = cert.Subject,
                ValidUntil = cert.NotAfter,
                Issuer = cert.Issuer,
                SerialNumber = cert.SerialNumber
            };
        }
        catch (System.Security.Cryptography.CryptographicException)
        {
            return CertificateLoadResult.Error("Senha incorreta ou certificado corrompido.");
        }
        catch (Exception ex)
        {
            return CertificateLoadResult.Error($"Erro ao carregar: {ex.Message}");
        }
    }

    /// <summary>
    /// Unload current certificate from memory.
    /// </summary>
    public void UnloadCertificate()
    {
        CurrentCertificate?.Dispose();
        CurrentCertificate = null;
    }
}

public class CertificateLoadResult
{
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public string? Subject { get; set; }
    public DateTime? ValidUntil { get; set; }
    public string? Issuer { get; set; }
    public string? SerialNumber { get; set; }

    public static CertificateLoadResult Error(string msg) =>
        new() { Success = false, ErrorMessage = msg };
}
