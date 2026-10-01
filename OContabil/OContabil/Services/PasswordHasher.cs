using System.Security.Cryptography;
using System.Text;

namespace OContabil.Services;

/// <summary>
/// Hash de senhas com PBKDF2 (HMAC-SHA256, 120 000 iterações, salt de 16 bytes,
/// derivação de 32 bytes). Formato armazenado:
///
///     pbkdf2-sha256$&lt;iter&gt;$&lt;saltBase64&gt;$&lt;hashBase64&gt;
///
/// Inclui detecção e migração automática de hashes SHA-256 herdados.
/// </summary>
public static class PasswordHasher
{
    private const int SaltBytes = 16;
    private const int HashBytes = 32;
    private const int Iterations = 120_000;
    private const string Algorithm = "pbkdf2-sha256";

    /// <summary>
    /// Gera um novo hash PBKDF2 para a senha informada.
    /// </summary>
    public static string Hash(string password)
    {
        if (string.IsNullOrEmpty(password))
            throw new ArgumentException("Senha vazia não é permitida.", nameof(password));

        var salt = RandomNumberGenerator.GetBytes(SaltBytes);
        var derived = Derive(password, salt, Iterations, HashBytes);

        return string.Join('$',
            Algorithm,
            Iterations.ToString(),
            Convert.ToBase64String(salt),
            Convert.ToBase64String(derived));
    }

    /// <summary>
    /// Verifica se a senha confere com o hash armazenado.
    /// Aceita tanto o formato moderno PBKDF2 quanto o legado SHA-256 puro.
    /// </summary>
    public static bool Verify(string password, string stored)
    {
        if (string.IsNullOrEmpty(stored) || string.IsNullOrEmpty(password))
            return false;

        if (stored.StartsWith(Algorithm + "$", StringComparison.Ordinal))
            return VerifyPbkdf2(password, stored);

        return VerifyLegacySha256(password, stored);
    }

    /// <summary>
    /// Retorna verdadeiro se o hash armazenado for do formato legado e precisar
    /// ser atualizado para PBKDF2.
    /// </summary>
    public static bool NeedsUpgrade(string stored)
    {
        return !string.IsNullOrEmpty(stored)
               && !stored.StartsWith(Algorithm + "$", StringComparison.Ordinal);
    }

    private static bool VerifyPbkdf2(string password, string stored)
    {
        var parts = stored.Split('$');
        if (parts.Length != 4) return false;
        if (!int.TryParse(parts[1], out var iter) || iter <= 0) return false;

        byte[] salt;
        byte[] expected;
        try
        {
            salt = Convert.FromBase64String(parts[2]);
            expected = Convert.FromBase64String(parts[3]);
        }
        catch (FormatException)
        {
            return false;
        }

        var actual = Derive(password, salt, iter, expected.Length);
        return CryptographicOperations.FixedTimeEquals(actual, expected);
    }

    private static bool VerifyLegacySha256(string password, string stored)
    {
        // Compatibilidade com hashes SHA-256 puros gerados pela versão original.
        var legacyBytes = SHA256.HashData(Encoding.UTF8.GetBytes(password));
        var legacyHex = Convert.ToHexString(legacyBytes).ToLowerInvariant();
        return CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(legacyHex),
            Encoding.UTF8.GetBytes(stored.ToLowerInvariant()));
    }

    private static byte[] Derive(string password, byte[] salt, int iterations, int length)
    {
        using var pbkdf2 = new Rfc2898DeriveBytes(
            password,
            salt,
            iterations,
            HashAlgorithmName.SHA256);
        return pbkdf2.GetBytes(length);
    }
}
