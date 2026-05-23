using System.Globalization;
using System.IO;
using System.Text;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services.Exports;

/// <summary>
/// Layout de importação do Domínio Sistemas (Lançamentos Contábeis padrão TXT).
/// Cada lançamento é uma linha pipe-delimitada com data, débito, crédito,
/// valor e histórico, conforme manual do Domínio Contábil.
/// </summary>
public sealed class DominioExportService
{
    private const string DebitAccount = "1.1.2.01";   // Bancos Conta Movimento (default)
    private const string CreditAccount = "2.1.1.01";  // Fornecedores

    public string Export(ExportRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            throw new ArgumentException("Caminho de saída obrigatório.", nameof(request));

        using var db = new AppDbContext();
        var docs = ExportRepository.Query(db, request);

        var sb = new StringBuilder();
        sb.Append("0000|OCONTABIL_EXPORT|").Append(DateTime.Now.ToString("ddMMyyyyHHmmss")).AppendLine("|");

        int lineCount = 0;
        foreach (var d in docs)
        {
            var value = ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor", "valor_principal");
            if (value <= 0) continue;

            var date = ExtractedValueParser.ParseDate(d.ExtractedJson, "data_emissao", "vencimento") ?? d.UploadedAt;

            sb.Append("0001|")
              .Append(date.ToString("ddMMyyyy")).Append('|')
              .Append(d.ChartOfAccount?.Code ?? DebitAccount).Append('|')
              .Append(CreditAccount).Append('|')
              .Append(FormatValue(value)).Append('|')
              .Append(History(d)).Append('|')
              .Append(d.Client?.Cnpj.Replace(".", "").Replace("/", "").Replace("-", "") ?? "").Append('|')
              .Append(d.Filename).AppendLine("|");
            lineCount++;
        }

        sb.Append("9999|").Append(lineCount.ToString("D6")).AppendLine("|");

        Directory.CreateDirectory(Path.GetDirectoryName(request.OutputPath)!);
        File.WriteAllText(request.OutputPath, sb.ToString(), Encoding.Latin1);

        AuditLogger.Write(null, "export.dominio", "Documents", docs.Count, request.OutputPath);
        return request.OutputPath;
    }

    private static string FormatValue(decimal value) =>
        value.ToString("F2", CultureInfo.InvariantCulture).Replace('.', ',');

    private static string History(Document d)
    {
        var emit = ExtractedValueParser.ReadField(d.ExtractedJson, "nome_emitente", "razao_social_emitente", "beneficiario");
        var numero = ExtractedValueParser.ReadField(d.ExtractedJson, "numero_nota");
        if (!string.IsNullOrEmpty(emit) && !string.IsNullOrEmpty(numero))
            return $"{d.DocumentType} {numero} - {emit}";
        if (!string.IsNullOrEmpty(emit))
            return $"{d.DocumentType} - {emit}";
        return $"{d.DocumentType} - {d.Filename}";
    }
}
