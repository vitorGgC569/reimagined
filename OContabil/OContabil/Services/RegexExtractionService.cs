using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OContabil.Services;

/// <summary>
/// Fallback determinístico baseado em expressões regulares para documentos
/// fiscais brasileiros. Não depende de Python nem de ONNX e funciona como
/// rede de segurança quando os motores principais estão indisponíveis.
/// </summary>
public static class RegexExtractionService
{
    private static readonly Regex CnpjRegex = new(@"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}", RegexOptions.Compiled);
    private static readonly Regex CpfRegex = new(@"\d{3}\.?\d{3}\.?\d{3}-?\d{2}", RegexOptions.Compiled);
    private static readonly Regex DateRegex = new(@"\b(\d{2}/\d{2}/\d{4})\b", RegexOptions.Compiled);
    private static readonly Regex MoneyRegex = new(@"R\$\s*([\d\.]+,\d{2})", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex AccessKeyRegex = new(@"\b\d{44}\b", RegexOptions.Compiled);
    private static readonly Regex BoletoLineRegex = new(@"\d{47,48}", RegexOptions.Compiled);

    public static GlinerResult Extract(string text, string docType, double threshold)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new GlinerResult { Success = false, Error = "Texto vazio." };

        var fields = docType switch
        {
            "Boleto" => ExtractBoleto(text),
            "Holerite" => ExtractHolerite(text),
            "ExtratoBancario" => ExtractExtrato(text),
            "DARF" => ExtractDarf(text),
            "DANFE" => ExtractDanfe(text),
            _ => ExtractNotaFiscal(text)
        };

        // Calcula confiança a partir da densidade de campos extraídos.
        var expected = ExpectedFieldsCount(docType);
        var coverage = expected == 0 ? 0.0 : (double)fields.Count / expected;
        var confidence = Math.Min(0.95, 0.55 + (coverage * 0.40));

        var groupKey = docType switch
        {
            "Boleto" => "boleto",
            "Holerite" => "holerite",
            "ExtratoBancario" => "extrato",
            "DARF" => "darf",
            "DANFE" => "danfe",
            _ => "nota_fiscal"
        };
        var wrapped = new Dictionary<string, Dictionary<string, object>> { [groupKey] = fields };
        var json = JsonSerializer.Serialize(wrapped);
        using var doc = JsonDocument.Parse(json);

        return new GlinerResult
        {
            Success = true,
            Model = "regex-fallback",
            Extraction = doc.RootElement.Clone(),
            AvgConfidence = Math.Round(confidence, 4),
            EntityCount = fields.Count,
            ThresholdUsed = threshold,
            OcrText = text.Length > 1500 ? text[..1500] + "..." : text,
            TextLength = text.Length,
            Note = "Extracao via regex (fallback)."
        };
    }

    private static int ExpectedFieldsCount(string docType) => docType switch
    {
        "Boleto" => 6,
        "Holerite" => 8,
        "ExtratoBancario" => 5,
        "DARF" => 6,
        "DANFE" => 9,
        _ => 7
    };

    private static Dictionary<string, object> ExtractNotaFiscal(string text)
    {
        var fields = new Dictionary<string, object>();

        var cnpjs = CnpjRegex.Matches(text).Select(m => m.Value).Distinct().ToList();
        if (cnpjs.Count > 0) AddField(fields, "cnpj_emitente", cnpjs[0], 0.85);
        if (cnpjs.Count > 1) AddField(fields, "cnpj_destinatario", cnpjs[1], 0.80);

        var key = AccessKeyRegex.Match(text);
        if (key.Success) AddField(fields, "chave_acesso", key.Value, 0.92);

        var date = DateRegex.Match(text);
        if (date.Success) AddField(fields, "data_emissao", date.Groups[1].Value, 0.78);

        var money = MoneyRegex.Matches(text);
        if (money.Count > 0)
            AddField(fields, "valor_total", money.Cast<Match>().Last().Groups[1].Value, 0.75);

        var numero = Regex.Match(text, @"(?:N[°º]\s*|Nota Fiscal[^\d]*)(\d{4,9})", RegexOptions.IgnoreCase);
        if (numero.Success) AddField(fields, "numero_nota", numero.Groups[1].Value, 0.70);

        return fields;
    }

    private static Dictionary<string, object> ExtractDanfe(string text)
    {
        var fields = ExtractNotaFiscal(text);
        var protocolo = Regex.Match(text, @"PROTOCOLO[^\d]*(\d{15,16})", RegexOptions.IgnoreCase);
        if (protocolo.Success) AddField(fields, "protocolo_autorizacao", protocolo.Groups[1].Value, 0.85);
        return fields;
    }

    private static Dictionary<string, object> ExtractBoleto(string text)
    {
        var fields = new Dictionary<string, object>();
        var digitsOnly = Regex.Replace(text, @"\D", "");
        var line = BoletoLineRegex.Match(digitsOnly);
        if (line.Success)
        {
            AddField(fields, "linha_digitavel", line.Value, 0.95);
            var info = BrasilApiService.ParseBoleto(line.Value);
            if (info != null)
            {
                if (info.Value > 0)
                    AddField(fields, "valor", info.Value.ToString("F2", CultureInfo.InvariantCulture), 0.95);
                if (info.DueDate.HasValue)
                    AddField(fields, "vencimento", info.DueDate.Value.ToString("dd/MM/yyyy"), 0.95);
                if (info.BankCode > 0)
                    AddField(fields, "banco", info.BankCode.ToString("D3"), 0.85);
            }
        }
        var cnpj = CnpjRegex.Match(text);
        if (cnpj.Success) AddField(fields, "cnpj_beneficiario", cnpj.Value, 0.80);
        return fields;
    }

    private static Dictionary<string, object> ExtractHolerite(string text)
    {
        var fields = new Dictionary<string, object>();

        var cnpj = CnpjRegex.Match(text);
        if (cnpj.Success) AddField(fields, "cnpj_empregador", cnpj.Value, 0.85);

        var cpf = CpfRegex.Match(text);
        if (cpf.Success) AddField(fields, "cpf_empregado", cpf.Value, 0.80);

        var competencia = Regex.Match(text, @"(\d{2}/\d{4})");
        if (competencia.Success) AddField(fields, "competencia", competencia.Groups[1].Value, 0.85);

        var valores = MoneyRegex.Matches(text).Select(m => m.Groups[1].Value).ToList();
        if (valores.Count >= 1) AddField(fields, "salario_base", valores[0], 0.70);
        if (valores.Count >= 2) AddField(fields, "total_proventos", valores[^2], 0.70);
        if (valores.Count >= 1) AddField(fields, "salario_liquido", valores[^1], 0.75);

        return fields;
    }

    private static Dictionary<string, object> ExtractExtrato(string text)
    {
        var fields = new Dictionary<string, object>();

        var banco = Regex.Match(text, @"\b(BANCO\s+[A-Z\s]+)\b", RegexOptions.IgnoreCase);
        if (banco.Success) AddField(fields, "banco", banco.Groups[1].Value.Trim(), 0.70);

        var agencia = Regex.Match(text, @"AG[EÊ]NCIA[:\s]*(\d{3,5})", RegexOptions.IgnoreCase);
        if (agencia.Success) AddField(fields, "agencia", agencia.Groups[1].Value, 0.80);

        var conta = Regex.Match(text, @"CONTA[:\s]*(\d{4,10}-?\d)", RegexOptions.IgnoreCase);
        if (conta.Success) AddField(fields, "conta", conta.Groups[1].Value, 0.80);

        var dates = DateRegex.Matches(text);
        if (dates.Count >= 2)
        {
            AddField(fields, "periodo_inicio", dates[0].Value, 0.75);
            AddField(fields, "periodo_fim", dates[^1].Value, 0.75);
        }
        var valores = MoneyRegex.Matches(text);
        if (valores.Count >= 2)
        {
            AddField(fields, "saldo_inicial", valores[0].Groups[1].Value, 0.65);
            AddField(fields, "saldo_final", valores[^1].Groups[1].Value, 0.65);
        }
        return fields;
    }

    private static Dictionary<string, object> ExtractDarf(string text)
    {
        var fields = new Dictionary<string, object>();
        var codigo = Regex.Match(text, @"C[ÓO]DIGO\s+DA\s+RECEITA[:\s]*(\d{4})", RegexOptions.IgnoreCase);
        if (codigo.Success) AddField(fields, "codigo_receita", codigo.Groups[1].Value, 0.90);
        var cnpj = CnpjRegex.Match(text);
        if (cnpj.Success) AddField(fields, "cnpj_contribuinte", cnpj.Value, 0.85);
        var date = DateRegex.Match(text);
        if (date.Success) AddField(fields, "data_vencimento", date.Groups[1].Value, 0.75);
        var valores = MoneyRegex.Matches(text);
        if (valores.Count >= 1) AddField(fields, "valor_principal", valores[0].Groups[1].Value, 0.75);
        if (valores.Count >= 2) AddField(fields, "valor_total", valores[^1].Groups[1].Value, 0.80);
        return fields;
    }

    private static void AddField(Dictionary<string, object> dict, string key, string value, double confidence)
    {
        dict[key] = new
        {
            text = value,
            confidence = Math.Round(confidence, 4)
        };
    }
}
