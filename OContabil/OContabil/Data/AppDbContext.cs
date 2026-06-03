using System.IO;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Data;

public class AppDbContext : DbContext
{
    public DbSet<User> Users { get; set; } = null!;
    public DbSet<Client> Clients { get; set; } = null!;
    public DbSet<Document> Documents { get; set; } = null!;
    public DbSet<ChartOfAccount> Accounts { get; set; } = null!;
    public DbSet<AuditLog> AuditLogs { get; set; } = null!;
    public DbSet<DocumentRevision> DocumentRevisions { get; set; } = null!;
    public DbSet<DocumentSchema> DocumentSchemas { get; set; } = null!;

    public static string DatabasePath { get; } = ResolveDbPath();

    private static string ResolveDbPath()
    {
        var dbPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "OContabil", "ocontabil.db");

        Directory.CreateDirectory(Path.GetDirectoryName(dbPath)!);
        return dbPath;
    }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        if (!options.IsConfigured)
        {
            // SQLCipher: banco inteiro cifrado em repouso. A chave (PRAGMA key) é
            // protegida por DPAPI (DbKey). Connstring montada via builder p/ escapar
            // corretamente a chave base64.
            var csb = new Microsoft.Data.Sqlite.SqliteConnectionStringBuilder
            {
                DataSource = DatabasePath,
                Cache = Microsoft.Data.Sqlite.SqliteCacheMode.Shared,
                Password = OContabil.Services.DbKey.Get(),
            };
            options.UseSqlite(csb.ConnectionString);
        }
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<User>()
            .HasIndex(u => u.Username)
            .IsUnique();

        modelBuilder.Entity<Client>()
            .HasIndex(c => c.Cnpj)
            .IsUnique();

        modelBuilder.Entity<Document>()
            .HasOne(d => d.Client)
            .WithMany()
            .HasForeignKey(d => d.ClientId)
            .OnDelete(DeleteBehavior.Restrict);

        modelBuilder.Entity<Document>()
            .HasOne(d => d.UploadedBy)
            .WithMany()
            .HasForeignKey(d => d.UploadedByUserId)
            .OnDelete(DeleteBehavior.Restrict);

        modelBuilder.Entity<Document>()
            .HasOne(d => d.ValidatedBy)
            .WithMany()
            .HasForeignKey(d => d.ValidatedByUserId)
            .OnDelete(DeleteBehavior.Restrict);

        modelBuilder.Entity<AuditLog>()
            .HasIndex(a => a.Timestamp);

        modelBuilder.Entity<DocumentRevision>()
            .HasIndex(r => r.DocumentId);

        modelBuilder.Entity<DocumentSchema>()
            .HasIndex(s => new { s.DocumentType, s.ClientId });

        // ── Cifragem em repouso (DPAPI) das colunas com PII fiscal ──
        // Transparente: cifra ao gravar, decifra ao ler. Dados legados em texto
        // são lidos como estão e re-cifrados no próximo save (migração preguiçosa).
        var encrypt = new ValueConverter<string?, string?>(
            v => Crypto.Protect(v),
            v => Crypto.Unprotect(v));
        modelBuilder.Entity<Document>().Property(d => d.ExtractedJson).HasConversion(encrypt);
        modelBuilder.Entity<Document>().Property(d => d.ExtractedJsonOriginal).HasConversion(encrypt);
        modelBuilder.Entity<Document>().Property(d => d.OcrText).HasConversion(encrypt);
    }
}
