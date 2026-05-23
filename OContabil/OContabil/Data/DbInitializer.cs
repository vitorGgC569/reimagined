using Microsoft.EntityFrameworkCore;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Data;

public static class DbInitializer
{
    public static void Initialize(AppDbContext db)
    {
        SchemaManager.Apply(db);

        db.Database.ExecuteSqlRaw("PRAGMA journal_mode=WAL;");
        db.Database.ExecuteSqlRaw("PRAGMA foreign_keys=ON;");

        SeedAccounts(db);
        SeedDefaultAdmin(db);
        SeedDocumentSchemas(db);
    }

    private static void SeedAccounts(AppDbContext db)
    {
        if (db.Accounts.Any()) return;

        db.Accounts.AddRange(
            new ChartOfAccount { Code = "3.1.1.01", Description = "Energia Elétrica", Type = "Despesa", AutoClassificationRegex = @"(?i)\b(enel|cemig|elektro|cpfl|light|copel|energia)\b" },
            new ChartOfAccount { Code = "3.1.1.02", Description = "Água e Esgoto", Type = "Despesa", AutoClassificationRegex = @"(?i)\b(sabesp|copasa|sanepar|cagece|saneago|dmae|agua|esgoto)\b" },
            new ChartOfAccount { Code = "3.1.1.03", Description = "Telefone e Internet", Type = "Despesa", AutoClassificationRegex = @"(?i)\b(vivo|claro|tim|oi|algar|telecom|fibra|internet)\b" },
            new ChartOfAccount { Code = "3.1.1.04", Description = "Material de Escritório", Type = "Despesa", AutoClassificationRegex = @"(?i)\b(kalunga|papelaria|materiais)\b" },
            new ChartOfAccount { Code = "3.1.1.05", Description = "Impostos e Taxas", Type = "Despesa", AutoClassificationRegex = @"(?i)\b(das|darf|gps|inss|fgts|iss|icms|tributos)\b" },
            new ChartOfAccount { Code = "3.1.1.06", Description = "Salários", Type = "Despesa", AutoClassificationRegex = @"(?i)\b(folha|salario|holerite|contracheque|rescisao)\b" },
            new ChartOfAccount { Code = "4.1.1.01", Description = "Receita de Serviços Prestados", Type = "Receita", AutoClassificationRegex = @"(?i)\b(prestacao\s*de\s*servicos|honorarios)\b" },
            new ChartOfAccount { Code = "1.1.2.01", Description = "Bancos Conta Movimento", Type = "Ativo", AutoClassificationRegex = @"(?i)\b(extrato|banco|saldo|deposito)\b" },
            new ChartOfAccount { Code = "3.2.1.01", Description = "Outras Despesas", Type = "Despesa", AutoClassificationRegex = null }
        );
        db.SaveChanges();
    }

    private static void SeedDefaultAdmin(AppDbContext db)
    {
        if (db.Users.Any()) return;

        db.Users.Add(new User
        {
            Username = "admin",
            PasswordHash = PasswordHasher.Hash("admin"),
            FullName = "Administrador",
            Role = UserRole.Admin,
            IsActive = true,
            MustChangePassword = true,
            CreatedAt = DateTime.UtcNow
        });
        db.SaveChanges();
    }

    private static void SeedDocumentSchemas(AppDbContext db)
    {
        if (db.DocumentSchemas.Any()) return;

        foreach (var schema in DocumentSchemaCatalog.BuiltIn())
            db.DocumentSchemas.Add(schema);

        db.SaveChanges();
    }
}
