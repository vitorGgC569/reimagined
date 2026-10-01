using System.IO;
using ClosedXML.Excel;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services.Exports;

/// <summary>
/// Exporta documentos validados para Excel (.xlsx) com formatação contábil,
/// abas separadas por cliente e sumário consolidado.
/// </summary>
public sealed class ExcelExportService
{
    public string Export(ExportRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            throw new ArgumentException("Caminho de saída obrigatório.", nameof(request));

        using var db = new AppDbContext();
        var docs = ExportRepository.Query(db, request);

        using var workbook = new XLWorkbook();
        BuildSummary(workbook, docs);
        BuildDetail(workbook, docs);
        BuildByClient(workbook, docs);

        Directory.CreateDirectory(Path.GetDirectoryName(request.OutputPath)!);
        workbook.SaveAs(request.OutputPath);

        AuditLogger.Write(null, "export.excel", "Documents", docs.Count, request.OutputPath);
        return request.OutputPath;
    }

    private static void BuildSummary(XLWorkbook wb, List<Document> docs)
    {
        var sheet = wb.AddWorksheet("Resumo");
        sheet.Cell("A1").Value = "OContabil — Relatório de Documentos";
        sheet.Cell("A1").Style.Font.Bold = true;
        sheet.Cell("A1").Style.Font.FontSize = 16;
        sheet.Range("A1:F1").Merge();

        sheet.Cell("A3").Value = "Gerado em:";
        sheet.Cell("B3").Value = DateTime.Now.ToString("dd/MM/yyyy HH:mm");
        sheet.Cell("A4").Value = "Total de documentos:";
        sheet.Cell("B4").Value = docs.Count;
        sheet.Cell("A5").Value = "Soma de valores:";
        sheet.Cell("B5").Value = docs.Sum(d => ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor"));
        sheet.Cell("B5").Style.NumberFormat.Format = "R$ #,##0.00";

        sheet.Cell("A7").Value = "Status";
        sheet.Cell("B7").Value = "Quantidade";
        sheet.Range("A7:B7").Style.Font.Bold = true;

        var statusGroups = docs.GroupBy(d => d.StatusDisplay).OrderByDescending(g => g.Count()).ToList();
        for (int i = 0; i < statusGroups.Count; i++)
        {
            sheet.Cell(8 + i, 1).Value = statusGroups[i].Key;
            sheet.Cell(8 + i, 2).Value = statusGroups[i].Count();
        }
        sheet.Columns().AdjustToContents();
    }

    private static void BuildDetail(XLWorkbook wb, List<Document> docs)
    {
        var sheet = wb.AddWorksheet("Documentos");
        var headers = new[]
        {
            "ID", "Arquivo", "Tipo", "Cliente", "CNPJ", "Data Upload", "Data Validação",
            "Confiança", "Conta", "Valor Total", "Vencimento", "Status", "Usuário"
        };
        for (int i = 0; i < headers.Length; i++)
        {
            var cell = sheet.Cell(1, i + 1);
            cell.Value = headers[i];
            cell.Style.Font.Bold = true;
            cell.Style.Fill.BackgroundColor = XLColor.LightGray;
        }

        int row = 2;
        foreach (var d in docs)
        {
            sheet.Cell(row, 1).Value = d.Id;
            sheet.Cell(row, 2).Value = d.Filename;
            sheet.Cell(row, 3).Value = d.DocumentType;
            sheet.Cell(row, 4).Value = d.Client?.Name ?? "";
            sheet.Cell(row, 5).Value = d.Client?.Cnpj ?? "";
            sheet.Cell(row, 6).Value = d.UploadedAt;
            sheet.Cell(row, 6).Style.DateFormat.Format = "dd/MM/yyyy HH:mm";
            sheet.Cell(row, 7).Value = d.ValidatedAt;
            sheet.Cell(row, 7).Style.DateFormat.Format = "dd/MM/yyyy HH:mm";
            sheet.Cell(row, 8).Value = d.ConfidenceScore ?? 0.0;
            sheet.Cell(row, 8).Style.NumberFormat.Format = "0.00%";
            sheet.Cell(row, 9).Value = d.ChartOfAccount?.Code + " " + d.ChartOfAccount?.Description;

            var value = ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor", "valor_principal", "salario_liquido");
            sheet.Cell(row, 10).Value = value;
            sheet.Cell(row, 10).Style.NumberFormat.Format = "R$ #,##0.00";

            var due = ExtractedValueParser.ParseDate(d.ExtractedJson, "vencimento", "data_vencimento", "data_emissao");
            if (due.HasValue)
            {
                sheet.Cell(row, 11).Value = due.Value;
                sheet.Cell(row, 11).Style.DateFormat.Format = "dd/MM/yyyy";
            }
            sheet.Cell(row, 12).Value = d.StatusDisplay;
            sheet.Cell(row, 13).Value = d.UploadedBy?.FullName ?? "";
            row++;
        }
        sheet.RangeUsed()?.SetAutoFilter();
        sheet.SheetView.FreezeRows(1);
        sheet.Columns().AdjustToContents();
    }

    private static void BuildByClient(XLWorkbook wb, List<Document> docs)
    {
        var sheet = wb.AddWorksheet("Por Cliente");
        sheet.Cell("A1").Value = "Cliente";
        sheet.Cell("B1").Value = "CNPJ";
        sheet.Cell("C1").Value = "Documentos";
        sheet.Cell("D1").Value = "Valor Total";
        sheet.Range("A1:D1").Style.Font.Bold = true;

        int row = 2;
        foreach (var grp in docs.GroupBy(d => d.Client?.Id ?? 0))
        {
            var client = grp.First().Client;
            sheet.Cell(row, 1).Value = client?.Name ?? "(sem cliente)";
            sheet.Cell(row, 2).Value = client?.Cnpj ?? "";
            sheet.Cell(row, 3).Value = grp.Count();
            sheet.Cell(row, 4).Value = grp.Sum(d => ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor"));
            sheet.Cell(row, 4).Style.NumberFormat.Format = "R$ #,##0.00";
            row++;
        }
        sheet.Columns().AdjustToContents();
    }
}
