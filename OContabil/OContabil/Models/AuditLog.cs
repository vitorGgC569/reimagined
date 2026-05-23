using System.ComponentModel.DataAnnotations;

namespace OContabil.Models;

public class AuditLog
{
    public int Id { get; set; }

    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    public int? UserId { get; set; }

    [Required, MaxLength(80)]
    public string Action { get; set; } = "";

    [MaxLength(80)]
    public string? Entity { get; set; }

    public int? EntityId { get; set; }

    [MaxLength(2000)]
    public string? Details { get; set; }

    [MaxLength(64)]
    public string? IpAddress { get; set; }
}
