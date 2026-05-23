namespace ContaDocAI.Models;

public class ExtractedField
{
    public string Value { get; set; } = "";
    public double Confidence { get; set; }
    public int SpanStart { get; set; }
    public int SpanEnd { get; set; }

    public string ConfidenceLevel => Confidence >= 0.9 ? "High" : Confidence >= 0.8 ? "Medium" : "Low";
    public int ConfidencePercent => (int)(Confidence * 100);
}

public class Document
{
    public string Id { get; set; } = "";
    public string Filename { get; set; } = "";
    public string ClientName { get; set; } = "";
    public int ClientId { get; set; }
    public string DocumentType { get; set; } = "";
    public DateTime UploadedAt { get; set; }
    public string Status { get; set; } = "ready_for_review";
    public string OcrText { get; set; } = "";
    public Dictionary<string, ExtractedField> ExtractedFields { get; set; } = new();
}

public class QueueItem
{
    public string Id { get; set; } = "";
    public string Filename { get; set; } = "";
    public string Size { get; set; } = "";
    public int FilesCount { get; set; }
    public string Status { get; set; } = "queued";
    public int Progress { get; set; }
}
