using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Data.Sqlite;
using OContabil.Data;

namespace OContabil.Services;

/// <summary>
/// Migra (uma vez) um banco SQLite LEGADO em texto-claro para SQLCipher. Usa
/// sqlcipher_export para reescrever cifrado, VALIDA a abertura com a chave e só
/// então troca o arquivo de forma atômica (File.Replace), APAGANDO em seguida o
/// backup plaintext com segurança (overwrite + delete) para não deixar PII em
/// claro no disco. Idempotente: se o DB já é cifrado (ou não existe), não faz nada.
/// </summary>
public static class DbCipherMigration
{
    public static void EnsureEncrypted()
    {
        var path = AppDbContext.DatabasePath;
        var bak = path + ".plain.bak";

        // Defesa: remove com segurança qualquer backup plaintext de uma migração
        // anterior (ex.: crash após o File.Replace, antes da limpeza). Como o
        // replace é atômico, se 'bak' existe então 'path' já está cifrado — apagar
        // o backup é sempre seguro.
        SecureDelete(bak);

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

        // Troca ATÔMICA (NTFS): path<-cifrado, bak<-plaintext antigo. A validação
        // acima garante que o cifrado abre, então não há perda de dados. Em seguida
        // o backup plaintext é apagado com segurança — nada de PII em claro fica.
        File.Replace(tmp, path, bak, ignoreMetadataErrors: true);
        SqliteConnection.ClearAllPools();
        SecureDelete(bak);
        SafeLog.Info("dbmigrate", "Banco migrado para SQLCipher (backup plaintext apagado com segurança).");
    }

    /// <summary>
    /// Aplica um restore pendente (ver <see cref="DbBackupService.StageRestore"/>) ANTES
    /// de abrir o banco: troca o .pending (já cifrado com o db.key local) pelo DB atual.
    /// </summary>
    public static void ApplyPendingRestore()
    {
        var pending = DbBackupService.PendingPath;
        if (!File.Exists(pending)) return;
        var path = AppDbContext.DatabasePath;
        try
        {
            using (var v = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = pending, Password = DbKey.Get(), Pooling = false }.ConnectionString))
            {
                v.Open();
                using var c = v.CreateCommand();
                c.CommandText = "SELECT count(*) FROM sqlite_master;";
                c.ExecuteScalar();
            }
            SqliteConnection.ClearAllPools();
            if (File.Exists(path))
            {
                var prev = path + ".prev";
                File.Replace(pending, path, prev, ignoreMetadataErrors: true);
                try { if (File.Exists(prev)) File.Delete(prev); } catch { }
            }
            else File.Move(pending, path);
            SqliteConnection.ClearAllPools();
            SafeLog.Info("dbrestore", "Restore aplicado a partir de backup por senha.");
        }
        catch (Exception ex)
        {
            SafeLog.Error("dbrestore", ex);
            try { if (File.Exists(pending)) File.Delete(pending); } catch { } // pendente inválido: descarta
        }
    }

    /// <summary>
    /// Sobrescreve o arquivo com bytes aleatórios e o remove (best-effort). Em SSD a
    /// sobrescrita não garante apagamento físico por causa de wear-leveling, mas
    /// elimina o arquivo plaintext acessível. Nunca lança.
    /// </summary>
    private static void SecureDelete(string path)
    {
        try
        {
            if (!File.Exists(path)) return;
            var len = new FileInfo(path).Length;
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.None))
            {
                var buf = new byte[81920];
                long written = 0;
                while (written < len)
                {
                    RandomNumberGenerator.Fill(buf);
                    int n = (int)Math.Min(buf.Length, len - written);
                    fs.Write(buf, 0, n);
                    written += n;
                }
                fs.Flush(flushToDisk: true);
            }
            File.Delete(path);
        }
        catch (Exception ex) { SafeLog.Error("securedelete", ex); }
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
