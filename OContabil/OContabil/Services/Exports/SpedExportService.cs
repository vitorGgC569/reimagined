using System.Globalization;
using System.IO;
using System.Text;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services.Exports;

/// <summary>
/// Gera o arquivo SPED Fiscal (EFD-ICMS/IPI) no formato simplificado exigido
/// pela Receita Federal. Inclui os registros essenciais:
///
/// <list type="bullet">
///   <item>0000 — Abertura</item>
///   <item>0001 — Abertura do bloco 0</item>
///   <item>0150 — Tabela de cadastro do participante</item>
///   <item>0990 — Encerramento bloco 0</item>
///   <item>C001 — Abertura bloco C</item>
///   <item>C100 — Documento NF-e</item>
///   <item>C990 — Encerramento bloco C</item>
///   <item>9001/9990/9999 — Encerramento</item>
/// </list>
///
/// O layout completo segue a ATO COTEPE/ICMS Nº 44/2018 (versão simplificada para MVP).
/// </summary>
public sealed class SpedExportService
{
    public string Export(ExportRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            throw new ArgumentException("Caminho de saída obrigatório.", nameof(request));

        using var db = new AppDbContext();
        var docs = ExportRepository.Query(db, request);
        if (docs.Count == 0)
            return "";

        var inicio = request.StartDate ?? docs.Min(d => d.UploadedAt);
        var fim = request.EndDate ?? docs.Max(d => d.UploadedAt);

        var sb = new StringBuilder();
        var totalLines = 0;

        // 0000 — Abertura do arquivo digital
        AppendLine(sb, ref totalLines, "0000",
            "017",                                              // versão do leiaute
            "0",                                                // 0=Original
            inicio.ToString("ddMMyyyy"),
            fim.ToString("ddMMyyyy"),
            Normalize(AppSettings.CompanyName, 100),
            OnlyDigits(AppSettings.CompanyCnpj).PadRight(14),
            "",                                                 // CPF
            "BR",
            "",                                                 // IE
            "",                                                 // COD_MUN
            "",                                                 // IM
            "",                                                 // SUFRAMA
            "0",                                                // IND_PERFIL
            "1"                                                 // IND_ATIV (industrial=0/outros=1)
        );

        // ─── Bloco 0 ────────────────────────────────────────────────────
        AppendLine(sb, ref totalLines, "0001", "0");

        // Participantes únicos
        var participants = docs
            .Where(d => d.Client != null)
            .Select(d => d.Client!)
            .GroupBy(c => c.Cnpj)
            .Select(g => g.First())
            .ToList();

        foreach (var p in participants)
        {
            AppendLine(sb, ref totalLines, "0150",
                $"P{p.Id:D6}",
                Normalize(p.Name, 100),
                "1058",                       // País Brasil
                OnlyDigits(p.Cnpj),
                "",                           // CPF
                "",                           // IE
                "",                           // COD_MUN
                "",                           // SUFRAMA
                "",                           // ENDERECO
                "",                           // NUM
                "",                           // COMPL
                ""                            // BAIRRO
            );
        }

        AppendLine(sb, ref totalLines, "0990", (totalLines + 1).ToString());
        var bloco0Lines = totalLines;

        // ─── Bloco C ────────────────────────────────────────────────────
        var bloco0End = totalLines;
        AppendLine(sb, ref totalLines, "C001", "0");

        foreach (var d in docs.Where(x => IsFiscal(x.DocumentType)))
        {
            var value = ExtractedValueParser.ParseMoney(d.ExtractedJson, "valor_total", "valor");
            var numero = ExtractedValueParser.ReadField(d.ExtractedJson, "numero_nota");
            var chave = ExtractedValueParser.ReadField(d.ExtractedJson, "chave_acesso");
            var dataEmissao = ExtractedValueParser.ParseDate(d.ExtractedJson, "data_emissao") ?? d.UploadedAt;

            AppendLine(sb, ref totalLines, "C100",
                "1",                                          // IND_OPER 0=Entrada, 1=Saída
                "1",                                          // IND_EMIT 0=Própria, 1=Terceiros
                d.Client != null ? $"P{d.Client.Id:D6}" : "",
                "55",                                         // MOD modelo NF-e
                "00",                                         // COD_SIT
                "1",                                          // SER série
                OnlyDigits(numero).PadLeft(9, '0'),
                OnlyDigits(chave).PadRight(44),
                dataEmissao.ToString("ddMMyyyy"),
                dataEmissao.ToString("ddMMyyyy"),
                FormatMoney(value),                           // VL_DOC
                "0",                                          // IND_PGTO
                FormatMoney(0),                               // VL_DESC
                FormatMoney(0),                               // VL_ABAT_NT
                FormatMoney(value),                           // VL_MERC
                "1",                                          // IND_FRT
                FormatMoney(0),                               // VL_FRT
                FormatMoney(0),                               // VL_SEG
                FormatMoney(0),                               // VL_OUT_DA
                FormatMoney(0),                               // VL_BC_ICMS
                FormatMoney(0),                               // VL_ICMS
                FormatMoney(0),                               // VL_BC_ICMS_ST
                FormatMoney(0),                               // VL_ICMS_ST
                FormatMoney(0),                               // VL_IPI
                FormatMoney(0),                               // VL_PIS
                FormatMoney(0),                               // VL_COFINS
                FormatMoney(0),                               // VL_PIS_ST
                FormatMoney(0)                                // VL_COFINS_ST
            );
        }

        AppendLine(sb, ref totalLines, "C990", (totalLines - bloco0End + 1).ToString());

        // ─── Bloco 9 (encerramento) ────────────────────────────────────
        AppendLine(sb, ref totalLines, "9001", "0");
        AppendLine(sb, ref totalLines, "9990", "");   // será atualizado abaixo
        AppendLine(sb, ref totalLines, "9999", (totalLines + 1).ToString());

        Directory.CreateDirectory(Path.GetDirectoryName(request.OutputPath)!);
        File.WriteAllText(request.OutputPath, sb.ToString(), Encoding.Latin1);

        AuditLogger.Write(null, "export.sped", "Documents", docs.Count, request.OutputPath);
        return request.OutputPath;
    }

    private static bool IsFiscal(string docType) =>
        docType is "NF-e" or "NFS-e" or "CT-e" or "DANFE";

    private static void AppendLine(StringBuilder sb, ref int counter, params string[] fields)
    {
        sb.Append('|');
        foreach (var f in fields)
            sb.Append(f).Append('|');
        sb.Append('\r').Append('\n');
        counter++;
    }

    private static string FormatMoney(decimal value) =>
        value.ToString("0.00", CultureInfo.InvariantCulture).Replace('.', ',');

    private static string OnlyDigits(string? input) =>
        new(input?.Where(char.IsDigit).ToArray() ?? Array.Empty<char>());

    private static string Normalize(string input, int maxLen)
    {
        var v = (input ?? "").Trim();
        return v.Length > maxLen ? v[..maxLen] : v;
    }
}
