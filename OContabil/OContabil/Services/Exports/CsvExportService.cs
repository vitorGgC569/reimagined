using System.Globalization;
using System.IO;
using System.Text;
using OContabil.Data;

namespace OContabil.Services.Exports;

/// <summary>
/// Exporta documentos em CSV (UTF-8 BOM, separador ;), compatível com Excel
/// em locale pt-BR.
/// </summary>
public sealed class CsvExportService
{
    public string Export(ExportRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            throw new ArgumentException("Caminho de saída obrigatório.", nameof(request));

        using var db = new AppDbContext();
        var docs = ExportRepository.Query(db, request);

        var culture = CultureInfo.GetCultureInfo("pt-BR");
        var sb = new StringBuilder();
        sb.Append('﻿'); // BOM para Excel reconhecer UTF-8

        sb.AppendLine("ID;Arquivo;Tipo;Cliente;CNPJ;Data Upload;Confiança;Valor;Vencimento;Status");

        foreach (var d in docs)
        {
            var value = ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor", "valor_principal");
            var due = ExtractedValueParser.ParseDate(d.ExtractedJson, "vencimento", "data_vencimento", "data_emissao");

            sb.Append(d.Id).Append(';')
              .Append(Escape(d.Filename)).Append(';')
              .Append(Escape(d.DocumentType)).Append(';')
              .Append(Escape(d.Client?.Name ?? "")).Append(';')
              .Append(Escape(d.Client?.Cnpj ?? "")).Append(';')
              .Append(d.UploadedAt.ToString("dd/MM/yyyy HH:mm")).Append(';')
              .Append((d.ConfidenceScore ?? 0).ToString("P2", culture)).Append(';')
              .Append(value.ToString("F2", culture)).Append(';')
              .Append(due?.ToString("dd/MM/yyyy") ?? "").Append(';')
              .AppendLine(Escape(d.StatusDisplay));
        }

        Directory.CreateDirectory(Path.GetDirectoryName(request.OutputPath)!);
        File.WriteAllText(request.OutputPath, sb.ToString(), new UTF8Encoding(true));

        AuditLogger.Write(null, "export.csv", "Documents", docs.Count, request.OutputPath);
        return request.OutputPath;
    }

    // Delega ao SecureCsv: além do quoting RFC-4180, previne CSV/Formula Injection
    // (CWE-1236) — células que começam com '=', '+', '-', '@' ou tab são neutralizadas
    // para não executarem como fórmula ao abrir no Excel/LibreOffice.
    private static string Escape(string value) => OContabil.Services.SecureCsv.Cell(value);
}
