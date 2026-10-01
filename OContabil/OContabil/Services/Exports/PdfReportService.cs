using System.IO;
using OContabil.Data;
using OContabil.Models;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;

namespace OContabil.Services.Exports;

/// <summary>
/// Relatório PDF de conferência. Apresenta cabeçalho com empresa, sumário
/// agregado e tabela de documentos. Útil para arquivar com o cliente final.
/// </summary>
public sealed class PdfReportService
{
    static PdfReportService()
    {
        QuestPDF.Settings.License = LicenseType.Community;
    }

    public string Export(ExportRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            throw new ArgumentException("Caminho de saída obrigatório.", nameof(request));

        using var db = new AppDbContext();
        var docs = ExportRepository.Query(db, request);

        var company = AppSettings.CompanyName;
        var companyCnpj = AppSettings.CompanyCnpj;
        var period = $"{request.StartDate?.ToString("dd/MM/yyyy") ?? "—"} até {request.EndDate?.ToString("dd/MM/yyyy") ?? "—"}";

        QuestPDF.Fluent.Document.Create(container =>
        {
            container.Page(page =>
            {
                page.Margin(30);
                page.Size(PageSizes.A4);
                page.DefaultTextStyle(t => t.FontSize(10));

                page.Header().Column(col =>
                {
                    col.Item().Text(company).FontSize(18).SemiBold();
                    if (!string.IsNullOrEmpty(companyCnpj))
                        col.Item().Text($"CNPJ: {companyCnpj}");
                    col.Item().Text("Relatório de Conferência de Documentos").FontSize(14).SemiBold();
                    col.Item().Text($"Período: {period}");
                    col.Item().Text($"Gerado em: {DateTime.Now:dd/MM/yyyy HH:mm}");
                });

                page.Content().PaddingVertical(10).Column(col =>
                {
                    col.Item().Text($"Total de documentos: {docs.Count}").SemiBold();
                    var totalValue = docs.Sum(d => ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor"));
                    col.Item().Text($"Soma de valores: R$ {totalValue:N2}");

                    col.Item().PaddingTop(10).Table(table =>
                    {
                        table.ColumnsDefinition(c =>
                        {
                            c.ConstantColumn(25);
                            c.RelativeColumn(3);
                            c.RelativeColumn(2);
                            c.RelativeColumn(2);
                            c.ConstantColumn(60);
                            c.ConstantColumn(70);
                        });

                        table.Header(header =>
                        {
                            header.Cell().Element(Th).Text("ID");
                            header.Cell().Element(Th).Text("Arquivo");
                            header.Cell().Element(Th).Text("Cliente");
                            header.Cell().Element(Th).Text("Tipo");
                            header.Cell().Element(Th).Text("Valor");
                            header.Cell().Element(Th).Text("Status");
                        });

                        foreach (var d in docs)
                        {
                            var value = ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor");
                            table.Cell().Element(Td).Text(d.Id.ToString());
                            table.Cell().Element(Td).Text(d.Filename);
                            table.Cell().Element(Td).Text(d.Client?.Name ?? "");
                            table.Cell().Element(Td).Text(d.DocumentType);
                            table.Cell().Element(Td).AlignRight().Text(value.ToString("N2"));
                            table.Cell().Element(Td).Text(d.StatusDisplay);
                        }
                    });
                });

                page.Footer().AlignCenter().Text(t =>
                {
                    t.Span("OContabil v1.0  •  página ");
                    t.CurrentPageNumber();
                    t.Span(" de ");
                    t.TotalPages();
                });
            });
        }).GeneratePdf(request.OutputPath);

        AuditLogger.Write(null, "export.pdf", "Documents", docs.Count, request.OutputPath);
        return request.OutputPath;

        static IContainer Th(IContainer c) => c.Background(Colors.Grey.Lighten3).Padding(4).BorderBottom(1).BorderColor(Colors.Grey.Medium);
        static IContainer Td(IContainer c) => c.Padding(4).BorderBottom(0.5f).BorderColor(Colors.Grey.Lighten2);
    }
}
