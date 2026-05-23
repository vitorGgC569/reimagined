namespace OContabil.Services.Exports;

/// <summary>
/// Parâmetros aplicáveis a qualquer exportação (Excel, CSV, SPED, PDF, Domínio).
/// Filtros são opcionais — quando ausentes a exportação considera todos os
/// documentos validados.
/// </summary>
public sealed class ExportRequest
{
    public DateTime? StartDate { get; set; }
    public DateTime? EndDate { get; set; }
    public int? ClientId { get; set; }
    public string? DocumentType { get; set; }
    public bool OnlyValidated { get; set; } = true;
    public string OutputPath { get; set; } = "";

    public static ExportRequest ForFile(string path) =>
        new() { OutputPath = path };
}
