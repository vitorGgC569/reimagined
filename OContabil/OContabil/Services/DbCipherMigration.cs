using System;
using System.IO;
using System.Text;
using Microsoft.Data.Sqlite;
using OContabil.Data;

namespace OContabil.Services;

/// <summary>
/// Migra (uma vez) um banco SQLite LEGADO em texto-claro para SQLCipher. Usa
/// sqlcipher_export para reescrever cifrado, VALIDA a abertura com a chave e só
/// então troca o arquivo, preservando um backup ".plain.bak". Idempotente: se o
/// DB já é cifrado (ou não existe), não faz nada.
/// </summary>
public static class DbCipherMigration
{
    public static void EnsureEncrypted()
    {
        var path = AppDbContext.DatabasePath;
        if (!File.Exists(path) || !IsPlaintext(path)) return; // novo (nasce cifrado) ou já cifrado

        var key = DbKey.Get();
        var tmp = path + ".enc";
        try { if (File.Exists(tmp)) File.Delete(tmp); } catch { }

        static string Esc(string s) => s.Replace("'", "''"); // SQL interno, valores app-controlados

        // Abre o plaintext (sem chave) e exporta cifrado para o tmp.
        using (var conn = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path, Pooling = false }.ConnectionString))
        {
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = $"ATTACH DATABASE '{Esc(tmp)}' AS enc KEY '{Esc(key)}'; SELECT sqlcipher_export('enc'); DETACH DATABASE enc;";
            cmd.ExecuteNonQuery();
        }
        SqliteConnection.ClearAllPools();

        // Valida que o cifrado realmente abre com a chave ANTES de substituir.
        using (var v = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = tmp, Password = key, Pooling = false }.ConnectionString))
        {
            v.Open();
            using var c = v.CreateCommand();
            c.CommandText = "SELECT count(*) FROM sqlite_master;";
            c.ExecuteScalar();
        }
        SqliteConnection.ClearAllPools();

        File.Copy(path, path + ".plain.bak", overwrite: true); // backup do legado (remover após validar em prod)
        File.Delete(path);
        File.Move(tmp, path);
        SafeLog.Info("dbmigrate", "Banco migrado para SQLCipher (backup .plain.bak preservado).");
    }

    private static bool IsPlaintext(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            var h = new byte[16];
            return fs.Read(h, 0, 16) == 16 && Encoding.ASCII.GetString(h).StartsWith("SQLite format 3");
        }
        catch { return false; }
    }
}
