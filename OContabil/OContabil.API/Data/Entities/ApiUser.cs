using System.ComponentModel.DataAnnotations;

namespace OContabil.API.Data.Entities;

public class ApiUser
{
    public int Id { get; set; }

    [Required, MaxLength(50)]
    public string Username { get; set; } = "";

    [Required, MaxLength(200)]
    public string FullName { get; set; } = "";

    [MaxLength(120)]
    public string? Email { get; set; }

    [Required]
    public string PasswordHash { get; set; } = "";

    [MaxLength(20)]
    public string Role { get; set; } = "Operator";

    public bool IsActive { get; set; } = true;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? LastLoginAt { get; set; }
    public int? TenantId { get; set; }
}

public class ApiClient
{
    public int Id { get; set; }

    [Required, MaxLength(200)]
    public string Name { get; set; } = "";

    [Required, MaxLength(20)]
    public string Cnpj { get; set; } = "";

    [MaxLength(120)]
    public string? Email { get; set; }

    [MaxLength(20)]
    public string? Phone { get; set; }

    [MaxLength(60)]
    public string? TaxRegime { get; set; }

    public bool IsActive { get; set; } = true;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public int? TenantId { get; set; }
}

public class ApiDocument
{
    public int Id { get; set; }

    [Required, MaxLength(300)]
    public string Filename { get; set; } = "";

    [MaxLength(50)]
    public string DocumentType { get; set; } = "";

    [MaxLength(20)]
    public string Status { get; set; } = "Pending";

    public double? ConfidenceScore { get; set; }
    public string? OcrText { get; set; }
    public string? ExtractedJson { get; set; }
    public long FileSizeBytes { get; set; }
    public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
    public DateTime? ProcessedAt { get; set; }

    public int ClientId { get; set; }
    public int UploadedByUserId { get; set; }
    public int? TenantId { get; set; }
}
