using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace OContabil.Services;

/// <summary>
/// Chave do banco SQLCipher, protegida por DPAPI (escopo do usuário Windows).
/// Gerada uma única vez (32 bytes aleatórios, base64) e guardada CIFRADA em
/// %LocalAppData%\OContabil\db.key — só a conta Windows atual a recupera. Assim,
/// o arquivo .db (e a chave) são inúteis em outra máquina/usuário ou em disco roubado.
/// </summary>
public static class DbKey
{
    private static readonly string Dir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OContabil");
    private static readonly string KeyFile = Path.Combine(Dir, "db.key");
    private static string? _cached;

    public static string Get()
    {
        if (_cached != null) return _cached;
        Directory.CreateDirectory(Dir);

        if (File.Exists(KeyFile) && OperatingSystem.IsWindows())
        {
            try
            {
                var raw = ProtectedData.Unprotect(File.ReadAllBytes(KeyFile), null, DataProtectionScope.CurrentUser);
                return _cached = Encoding.UTF8.GetString(raw);
            }
            catch (Exception ex) { SafeLog.Error("dbkey.read", ex); }
        }

        var key = Convert.ToBase64String(RandomNumberGenerator.GetBytes(32));
        try
        {
            if (OperatingSystem.IsWindows())
                File.WriteAllBytes(KeyFile, ProtectedData.Protect(Encoding.UTF8.GetBytes(key), null, DataProtectionScope.CurrentUser));
        }
        catch (Exception ex) { SafeLog.Error("dbkey.write", ex); }
        return _cached = key;
    }
}
