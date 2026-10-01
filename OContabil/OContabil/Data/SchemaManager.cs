using Microsoft.EntityFrameworkCore;

namespace OContabil.Data;

/// <summary>
/// Controle de versões do schema SQLite. Cada execução verifica em ordem se há
/// upgrades a aplicar, registrando o numero em <c>SchemaVersion</c>. As migrations
/// são idempotentes e podem ser executadas várias vezes sem efeito colateral.
/// </summary>
public static class SchemaManager
{
    private const int CurrentVersion = 5;

    public static void Apply(AppDbContext db)
    {
        db.Database.EnsureCreated();
        CreateVersionTable(db);

        var version = ReadCurrentVersion(db);

        if (version < 1) ApplyV1(db);
        if (version < 2) ApplyV2(db);
        if (version < 3) ApplyV3(db);
        if (version < 4) ApplyV4(db);
        if (version < 5) ApplyV5(db);

        WriteCurrentVersion(db, CurrentVersion);
    }

    private static void CreateVersionTable(AppDbContext db)
    {
        Exec(db, @"CREATE TABLE IF NOT EXISTS ""SchemaVersion"" (
            ""Id"" INTEGER PRIMARY KEY,
            ""Version"" INTEGER NOT NULL,
            ""AppliedAt"" TEXT NOT NULL
        );");
    }

    private static int ReadCurrentVersion(AppDbContext db)
    {
        try
        {
            using var conn = db.Database.GetDbConnection();
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT IFNULL(MAX(Version), 0) FROM SchemaVersion";
            return Convert.ToInt32(cmd.ExecuteScalar() ?? 0);
        }
        catch { return 0; }
    }

    private static void WriteCurrentVersion(AppDbContext db, int version)
    {
        // Parametrizado (sem interpolação em SQL) — rigor anti-injection mesmo
        // sendo valores internos/confiáveis.
        try { db.Database.ExecuteSqlRaw("INSERT INTO SchemaVersion(Version, AppliedAt) VALUES ({0}, {1})", version, DateTime.UtcNow.ToString("O")); }
        catch { }
    }

    // V1 — auditoria, revisões e schemas configuráveis.
    private static void ApplyV1(AppDbContext db)
    {
        Exec(db, @"CREATE TABLE IF NOT EXISTS ""AuditLogs"" (
            ""Id"" INTEGER NOT NULL CONSTRAINT ""PK_AuditLogs"" PRIMARY KEY AUTOINCREMENT,
            ""Timestamp"" TEXT NOT NULL,
            ""UserId"" INTEGER NULL,
            ""Action"" TEXT NOT NULL,
            ""Entity"" TEXT NULL,
            ""EntityId"" INTEGER NULL,
            ""Details"" TEXT NULL,
            ""IpAddress"" TEXT NULL
        );");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_AuditLogs_Timestamp ON AuditLogs(Timestamp);");

        Exec(db, @"CREATE TABLE IF NOT EXISTS ""DocumentRevisions"" (
            ""Id"" INTEGER NOT NULL CONSTRAINT ""PK_DocumentRevisions"" PRIMARY KEY AUTOINCREMENT,
            ""DocumentId"" INTEGER NOT NULL,
            ""UserId"" INTEGER NULL,
            ""RevisedAt"" TEXT NOT NULL,
            ""PreviousJson"" TEXT NULL,
            ""NewJson"" TEXT NULL,
            ""PreviousStatus"" INTEGER NOT NULL,
            ""NewStatus"" INTEGER NOT NULL,
            ""Reason"" TEXT NULL
        );");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_DocumentRevisions_DocumentId ON DocumentRevisions(DocumentId);");

        Exec(db, @"CREATE TABLE IF NOT EXISTS ""DocumentSchemas"" (
            ""Id"" INTEGER NOT NULL CONSTRAINT ""PK_DocumentSchemas"" PRIMARY KEY AUTOINCREMENT,
            ""Name"" TEXT NOT NULL,
            ""DocumentType"" TEXT NOT NULL,
            ""ClientId"" INTEGER NULL,
            ""SchemaJson"" TEXT NOT NULL,
            ""IsSystem"" INTEGER NOT NULL DEFAULT 0,
            ""CreatedAt"" TEXT NOT NULL
        );");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_DocumentSchemas_Type ON DocumentSchemas(DocumentType, ClientId);");
    }

    // V2 — campos extras em Documents.
    private static void ApplyV2(AppDbContext db)
    {
        Exec(db, "ALTER TABLE Documents ADD COLUMN ExtractedJsonOriginal TEXT NULL;");
        Exec(db, "ALTER TABLE Documents ADD COLUMN RejectionReason TEXT NULL;");
        Exec(db, "ALTER TABLE Documents ADD COLUMN LastEditedAt TEXT NULL;");
    }

    // V3 — campos extras em Users.
    private static void ApplyV3(AppDbContext db)
    {
        Exec(db, "ALTER TABLE Users ADD COLUMN LastLoginAt TEXT NULL;");
        Exec(db, "ALTER TABLE Users ADD COLUMN Email TEXT NULL;");
        Exec(db, "ALTER TABLE Users ADD COLUMN MustChangePassword INTEGER NOT NULL DEFAULT 0;");
    }

    // V4 — garante FilePath e ChartOfAccountId em Documents (compat).
    private static void ApplyV4(AppDbContext db)
    {
        Exec(db, "ALTER TABLE Documents ADD COLUMN FilePath TEXT NULL;");
        Exec(db, "ALTER TABLE Documents ADD COLUMN ChartOfAccountId INTEGER NULL;");
    }

    // V5 — valor numérico indexável + índices de consulta em Documents
    // (consulta rápida por valor/tipo/cliente/data — "consulta por índice").
    private static void ApplyV5(AppDbContext db)
    {
        Exec(db, "ALTER TABLE Documents ADD COLUMN ValorTotal REAL NULL;");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_Documents_Client ON Documents(ClientId);");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_Documents_Type ON Documents(DocumentType);");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_Documents_Uploaded ON Documents(UploadedAt);");
        Exec(db, "CREATE INDEX IF NOT EXISTS IX_Documents_Valor ON Documents(ValorTotal);");
    }

    private static void Exec(AppDbContext db, string sql)
    {
        try { db.Database.ExecuteSqlRaw(sql); } catch { /* idempotente */ }
    }
}
