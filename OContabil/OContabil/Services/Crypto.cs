using System;
using System.Security.Cryptography;
using System.Text;

namespace OContabil.Services;

/// <summary>
/// Cifragem de dados sensíveis EM REPOUSO via Windows DPAPI (escopo do usuário
/// atual). O texto cifrado só é decifrável pela MESMA conta Windows nesta
/// máquina — protege a PII fiscal (JSON extraído, texto OCR) contra roubo do
/// arquivo de banco, cópia de disco e outros usuários do SO, sem gerência
/// manual de chave. Tolera dados legados: texto sem o marcador é lido como está
/// e passa a ser cifrado no próximo save (migração preguiçosa, sem quebra).
/// </summary>
public static class Crypto
{
    private const string Marker = "dpapi:v1:";
    // Entropia adicional específica do app (camada extra além da chave do usuário).
    private static readonly byte[] Entropy = Encoding.UTF8.GetBytes("OContabil.LocalData.v1");

    /// <summary>Cifra (idempotente p/ null/vazio; nunca lança — falha = devolve claro).</summary>
    public static string? Protect(string? plaintext)
    {
        if (string.IsNullOrEmpty(plaintext)) return plaintext;
        if (IsEncrypted(plaintext)) return plaintext; // já cifrado
        try
        {
            if (!OperatingSystem.IsWindows()) return plaintext;
            var enc = ProtectedData.Protect(Encoding.UTF8.GetBytes(plaintext), Entropy, DataProtectionScope.CurrentUser);
            return Marker + Convert.ToBase64String(enc);
        }
        catch { return plaintext; }
    }

    /// <summary>Decifra; texto legado (sem marcador) é devolvido inalterado.</summary>
    public static string? Unprotect(string? stored)
    {
        if (string.IsNullOrEmpty(stored) || !stored.StartsWith(Marker, StringComparison.Ordinal))
            return stored;
        try
        {
            if (!OperatingSystem.IsWindows()) return stored;
            var raw = Convert.FromBase64String(stored.Substring(Marker.Length));
            var dec = ProtectedData.Unprotect(raw, Entropy, DataProtectionScope.CurrentUser);
            return Encoding.UTF8.GetString(dec);
        }
        catch { return stored; }
    }

    public static bool IsEncrypted(string? stored) =>
        !string.IsNullOrEmpty(stored) && stored.StartsWith(Marker, StringComparison.Ordinal);
}
