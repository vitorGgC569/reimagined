using System.IO;
using System.Security.Cryptography;
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
        SeedDemoClients(db);
    }

    // Clientes de demonstração — apenas se a base estiver vazia (não-destrutivo).
    // Dá conteúdo real às telas de Clientes/Documentos ao apresentar a UI nova.
    private static void SeedDemoClients(AppDbContext db)
    {
        if (db.Clients.Any()) return;

        db.Clients.AddRange(
            new Client { Name = "Marília Comércio de Alimentos Ltda", Cnpj = "12.345.678/0001-90", TaxRegime = "Simples Nacional", DocumentCount = 482, ValidatedCount = 430 },
            new Client { Name = "Construtora Horizonte Norte S/A", Cnpj = "08.776.443/0001-55", TaxRegime = "Lucro Real", DocumentCount = 367, ValidatedCount = 310 },
            new Client { Name = "Drogaria São Lucas Eireli", Cnpj = "23.998.112/0001-07", TaxRegime = "Simples Nacional", DocumentCount = 311, ValidatedCount = 298 },
            new Client { Name = "Transportes Vale do Aço Ltda", Cnpj = "31.554.700/0001-21", TaxRegime = "Lucro Presumido", DocumentCount = 298, ValidatedCount = 240 },
            new Client { Name = "Estúdio Verde Arquitetura ME", Cnpj = "40.221.889/0001-44", TaxRegime = "Simples Nacional", DocumentCount = 142, ValidatedCount = 120 },
            new Client { Name = "Padaria e Confeitaria Pão Dourado Ltda", Cnpj = "55.013.226/0001-18", TaxRegime = "Simples Nacional", DocumentCount = 256, ValidatedCount = 233 }
        );
        db.SaveChanges();
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

        // Segurança: NUNCA embarcar uma credencial fixa (admin/admin). Gera uma
        // senha temporária aleatória forte e a grava em FIRST_ACCESS.txt no perfil
        // local — só quem tem acesso à máquina a lê. MustChangePassword força a
        // troca no primeiro login.
        var temporaryPassword = GenerateTemporaryPassword();
        db.Users.Add(new User
        {
            Username = "admin",
            PasswordHash = PasswordHasher.Hash(temporaryPassword),
            FullName = "Administrador",
            Role = UserRole.Admin,
            IsActive = true,
            MustChangePassword = true,
            CreatedAt = DateTime.UtcNow
        });
        db.SaveChanges();

        PersistFirstAccessInfo(temporaryPassword);
    }

    private static string GenerateTemporaryPassword()
    {
        const string alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%";
        var buffer = RandomNumberGenerator.GetBytes(16);
        return new string(buffer.Select(b => alphabet[b % alphabet.Length]).ToArray());
    }

    private static void PersistFirstAccessInfo(string temporaryPassword)
    {
        try
        {
            var root = Path.GetDirectoryName(AppDbContext.DatabasePath)!;
            Directory.CreateDirectory(root);
            var content = string.Join(Environment.NewLine,
                "OContabil - Primeiro acesso",
                $"Gerado em: {DateTime.Now:yyyy-MM-dd HH:mm:ss}",
                "Usuario: admin",
                $"Senha temporaria: {temporaryPassword}",
                "Altere essa credencial logo apos o primeiro login (o app vai exigir).");
            File.WriteAllText(Path.Combine(root, "FIRST_ACCESS.txt"), content, System.Text.Encoding.UTF8);
        }
        catch (Exception ex)
        {
            SafeLog.Error("firstaccess.write", ex);
        }
    }

    private static void SeedDocumentSchemas(AppDbContext db)
    {
        if (db.DocumentSchemas.Any()) return;

        foreach (var schema in DocumentSchemaCatalog.BuiltIn())
            db.DocumentSchemas.Add(schema);

        db.SaveChanges();
    }
}
