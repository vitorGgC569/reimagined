namespace OContabil.Services.Exports;

public enum ExportFormat
{
    Excel,
    Csv,
    Sped,
    Dominio,
    Pdf
}

/// <summary>
/// Fachada simples para todos os formatos de exportação. Centraliza nomes de
/// arquivo padrão e despacha para o service correto.
/// </summary>
public sealed class ExportManager
{
    public string Export(ExportFormat format, ExportRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            request.OutputPath = DefaultPath(format);

        var path = format switch
        {
            ExportFormat.Excel => new ExcelExportService().Export(request),
            ExportFormat.Csv => new CsvExportService().Export(request),
            ExportFormat.Sped => new SpedExportService().Export(request),
            ExportFormat.Dominio => new DominioExportService().Export(request),
            ExportFormat.Pdf => new PdfReportService().Export(request),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Formato não suportado")
        };
        if (!string.IsNullOrEmpty(path))
            OContabil.Services.TelemetryService.IncrementExport();
        return path;
    }

    public string DefaultExtension(ExportFormat format) => format switch
    {
        ExportFormat.Excel => "xlsx",
        ExportFormat.Csv => "csv",
        ExportFormat.Sped => "txt",
        ExportFormat.Dominio => "txt",
        ExportFormat.Pdf => "pdf",
        _ => "out"
    };

    public string DefaultFilter(ExportFormat format) => format switch
    {
        ExportFormat.Excel => "Planilha Excel (*.xlsx)|*.xlsx",
        ExportFormat.Csv => "CSV (*.csv)|*.csv",
        ExportFormat.Sped => "SPED (*.txt)|*.txt",
        ExportFormat.Dominio => "Domínio TXT (*.txt)|*.txt",
        ExportFormat.Pdf => "PDF (*.pdf)|*.pdf",
        _ => "Arquivo|*.*"
    };

    private string DefaultPath(ExportFormat format)
    {
        var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var name = $"OContabil_{format}_{stamp}.{DefaultExtension(format)}";
        return System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), name);
    }
}
