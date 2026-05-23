namespace OContabil.Models;

public class DocumentRevision
{
    public int Id { get; set; }
    public int DocumentId { get; set; }
    public int? UserId { get; set; }
    public DateTime RevisedAt { get; set; } = DateTime.UtcNow;
    public string? PreviousJson { get; set; }
    public string? NewJson { get; set; }
    public DocumentStatus PreviousStatus { get; set; }
    public DocumentStatus NewStatus { get; set; }
    public string? Reason { get; set; }
}
