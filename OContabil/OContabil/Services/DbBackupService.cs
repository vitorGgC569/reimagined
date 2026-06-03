using System;
using System.IO;
using Microsoft.Data.Sqlite;
using OContabil.Data;

namespace OContabil.Services;

/// <summary>
/// Backup/recuperação cifrados por SENHA do usuário, INDEPENDENTE do db.key/DPAPI.
/// Resolve o cenário em que o db.key é perdido (reinstalação do Windows, troca de
/// conta, perfil corrompido) — sem isso o banco SQLCipher seria irrecuperável. O
/// .ocbak é um banco SQLCipher cifrado com a senha; o restore re-cifra com o db.key
/// LOCAL e é aplicado no próximo startup (evita travar o arquivo em uso).
/// </summary>
public static class DbBackupService
{
    private static string Esc(string s) => s.Replace("'", "''"); // SQL interno (senha/caminho app-controlados)

    public static string PendingPath => AppDbContext.DatabasePath + ".pending";

    /// <summary>Exporta a DB atual para um .ocbak cifrado com a senha informada.</summary>
    public static void Export(string destPath, string password)
    {
        if (string.IsNullOrWhiteSpace(password) || password.Length < 6)
            throw new ArgumentException("A senha do backup deve ter ao menos 6 caracteres.");
        var src = AppDbContext.DatabasePath;
        if (!File.Exists(src)) throw new FileNotFoundException("Banco de dados não encontrado.");
        try { if (File.Exists(destPath)) File.Delete(destPath); } catch { }

        using var conn = new SqliteConnection(new SqliteConnectionStringBuilder
        { DataSource = src, Password = DbKey.Get(), Pooling = false }.ConnectionString);
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $"ATTACH DATABASE '{Esc(destPath)}' AS bak KEY '{Esc(password)}'; SELECT sqlcipher_export('bak'); DETACH DATABASE bak;";
        cmd.ExecuteNonQuery();
        SqliteConnection.ClearAllPools();
        SafeLog.Info("dbbackup", "Backup cifrado por senha exportado.");
    }

    /// <summary>
    /// Valida a senha do .ocbak, re-cifra com o db.key LOCAL e deixa pronto em
    /// <see cref="PendingPath"/>. A troca acontece no próximo startup
    /// (<see cref="DbCipherMigration.ApplyPendingRestore"/>). Lança se a senha estiver errada.
    /// </summary>
    public static void StageRestore(string srcBackup, string password)
    {
        if (!File.Exists(srcBackup)) throw new FileNotFoundException("Arquivo de backup não encontrado.");
        var tmp = PendingPath;
        try { if (File.Exists(tmp)) File.Delete(tmp); } catch { }

        // Abre o backup com a senha (lança se incorreta) e re-exporta cifrado com o db.key local.
        using (var conn = new SqliteConnection(new SqliteConnectionStringBuilder
        { DataSource = srcBackup, Password = password, Pooling = false }.ConnectionString))
        {
            conn.Open();
            using var probe = conn.CreateCommand();
            probe.CommandText = "SELECT count(*) FROM sqlite_master;"; // confirma senha/integridade
            probe.ExecuteScalar();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = $"ATTACH DATABASE '{Esc(tmp)}' AS loc KEY '{Esc(DbKey.Get())}'; SELECT sqlcipher_export('loc'); DETACH DATABASE loc;";
            cmd.ExecuteNonQuery();
        }
        SqliteConnection.ClearAllPools();

        // Valida que o pendente abre com o db.key local.
        using (var v = new SqliteConnection(new SqliteConnectionStringBuilder
        { DataSource = tmp, Password = DbKey.Get(), Pooling = false }.ConnectionString))
        {
            v.Open();
            using var c = v.CreateCommand();
            c.CommandText = "SELECT count(*) FROM sqlite_master;";
            c.ExecuteScalar();
        }
        SqliteConnection.ClearAllPools();
        SafeLog.Info("dbbackup", "Restore preparado; será aplicado no próximo início.");
    }
}
