using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OContabil.Models;

public enum DocumentStatus
{
    Pending,
    Processing,
    ReadyForReview,
    Validated,
    Rejected,
    Error
}

public class Document
{
    public int Id { get; set; }

    [Required, MaxLength(300)]
    public string Filename { get; set; } = "";

    [MaxLength(1000)]
    public string? FilePath { get; set; }

    [MaxLength(50)]
    public string DocumentType { get; set; } = "";

    public DocumentStatus Status { get; set; } = DocumentStatus.Pending;

    public double? ConfidenceScore { get; set; }

    /// <summary>Valor total numérico extraído (R$), indexado para consulta. Null se não houver.</summary>
    public double? ValorTotal { get; set; }

    public string? OcrText { get; set; }

    public string? ExtractedJson { get; set; }

    /// <summary>JSON original retornado pela IA (preservado mesmo após edição manual).</summary>
    public string? ExtractedJsonOriginal { get; set; }

    public long FileSizeBytes { get; set; }

    public DateTime UploadedAt { get; set; } = DateTime.Now;

    public DateTime? ProcessedAt { get; set; }

    public DateTime? ValidatedAt { get; set; }

    public DateTime? LastEditedAt { get; set; }

    [MaxLength(500)]
    public string? RejectionReason { get; set; }

    public int ClientId { get; set; }
    [ForeignKey("ClientId")]
    public Client? Client { get; set; }

    public int? ChartOfAccountId { get; set; }
    [ForeignKey("ChartOfAccountId")]
    public ChartOfAccount? ChartOfAccount { get; set; }

    public int UploadedByUserId { get; set; }
    [ForeignKey("UploadedByUserId")]
    public User? UploadedBy { get; set; }

    public int? ValidatedByUserId { get; set; }
    [ForeignKey("ValidatedByUserId")]
    public User? ValidatedBy { get; set; }

    public string StatusDisplay => Status switch
    {
        DocumentStatus.Pending => "Pendente",
        DocumentStatus.Processing => "Processando",
        DocumentStatus.ReadyForReview => "Revisar",
        DocumentStatus.Validated => "Validado",
        DocumentStatus.Rejected => "Rejeitado",
        DocumentStatus.Error => "Erro",
        _ => "Desconhecido"
    };

    public string FileSizeDisplay
    {
        get
        {
            if (FileSizeBytes < 1024) return $"{FileSizeBytes} B";
            if (FileSizeBytes < 1024 * 1024) return $"{FileSizeBytes / 1024.0:F1} KB";
            return $"{FileSizeBytes / (1024.0 * 1024.0):F1} MB";
        }
    }
}
