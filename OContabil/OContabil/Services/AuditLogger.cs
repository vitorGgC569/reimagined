using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services;

/// <summary>
/// API simples para gravação de entradas em <see cref="AuditLog"/>.
/// As escritas são sempre best-effort e nunca lançam exceção, para não
/// derrubar fluxos de negócio.
/// </summary>
public static class AuditLogger
{
    public static void Write(int? userId, string action, string? entity = null, int? entityId = null, string? details = null)
    {
        try
        {
            using var db = new AppDbContext();
            Write(db, userId, action, entity, entityId, details);
        }
        catch { }
    }

    public static void Write(AppDbContext db, int? userId, string action, string? entity = null, int? entityId = null, string? details = null)
    {
        try
        {
            db.AuditLogs.Add(new AuditLog
            {
                Timestamp = DateTime.UtcNow,
                UserId = userId,
                Action = action,
                Entity = entity,
                EntityId = entityId,
                Details = Truncate(details, 2000)
            });
            db.SaveChanges();
        }
        catch { }
    }

    private static string? Truncate(string? value, int max)
    {
        if (string.IsNullOrEmpty(value)) return value;
        return value.Length <= max ? value : value[..max];
    }
}
