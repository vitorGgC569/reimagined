using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OContabil.Services;

/// <summary>
/// Extrator determinístico (rótulo + padrão) para documentos fiscais brasileiros.
/// Não depende de Python nem de ONNX. Combina busca por rótulo ("Valor: R$ ...")
/// com padrões fortes (CNPJ validado, R$, datas, chave 44 díg, linha digitável)
/// por tipo de documento. Toda a extração é guardada — nunca lança — de modo que
/// um documento legível nunca cai em "Erro" por falha do extrator.
/// </summary>
public static class RegexExtractionService
{
    private const RegexOptions O = RegexOptions.Compiled | RegexOptions.IgnoreCase;

    private static readonly Regex CnpjRx = new(@"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}", RegexOptions.Compiled);
    private static readonly Regex CpfRx = new(@"\d{3}\.?\d{3}\.?\d{3}-?\d{2}", RegexOptions.Compiled);
    private static readonly Regex DateRx = new(@"\b(\d{2}/\d{2}/\d{4})\b", RegexOptions.Compiled);
    private static readonly Regex CompetenciaRx = new(@"\b(\d{2}/\d{4})\b", RegexOptions.Compiled);
    private static readonly Regex MoneyRx = new(@"R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})", O);
    private static readonly Regex MoneyBareRx = new(@"\b(\d{1,3}(?:\.\d{3})*,\d{2})\b", RegexOptions.Compiled);
    private static readonly Regex AccessKeyRx = new(@"\b\d{44}\b", RegexOptions.Compiled);
    // Linha digitável formatada (5.5 5.6 5.6 1 14) — mais confiável que dígitos crus.
    private static readonly Regex LinhaFmtRx = new(@"\d{5}\.\d{5}\s+\d{5}\.\d{6}\s+\d{5}\.\d{6}\s+\d\s+\d{14}", RegexOptions.Compiled);

    public static GlinerResult Extract(string text, string docType, double threshold)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new GlinerResult { Success = false, Error = "Texto vazio." };

        Dictionary<string, object> fields;
        try
        {
            fields = docType switch
            {
                "Boleto" => ExtractBoleto(text),
                "Holerite" => ExtractHolerite(text),
                "ExtratoBancario" => ExtractExtrato(text),
                "DARF" => ExtractDarf(text),
                "DANFE" => ExtractDanfe(text),
                _ => ExtractNotaFiscal(text)
            };
        }
        catch
        {
            // Nunca falhar por causa do extrator — devolve genérico do que der.
            fields = ExtractGeneric(text);
        }

        if (fields.Count == 0) fields = ExtractGeneric(text);

        var expected = ExpectedFieldsCount(docType);
        var coverage = expected == 0 ? 0.0 : Math.Min(1.0, (double)fields.Count / expected);
        var confidence = Math.Round(Math.Min(0.95, 0.50 + coverage * 0.45), 4);

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
            AvgConfidence = confidence,
            EntityCount = fields.Count,
            ThresholdUsed = threshold,
            OcrText = text.Length > 1500 ? text[..1500] + "..." : text,
            TextLength = text.Length,
            Note = "Extracao deterministica (rotulo+padrao)."
        };
    }

    private static int ExpectedFieldsCount(string docType) => docType switch
    {
        "Boleto" => 6,
        "Holerite" => 8,
        "ExtratoBancario" => 6,
        "DARF" => 6,
        "DANFE" => 9,
        _ => 7
    };

    // ── Extractors por tipo ───────────────────────────────────────────────

    private static Dictionary<string, object> ExtractBoleto(string text)
    {
        var f = new Dictionary<string, object>();

        // Linha digitável: preferir a formatada; senão a linha rotulada.
        var linha = LinhaFmtRx.Match(text);
        string? linhaDig = linha.Success ? Regex.Replace(linha.Value, @"\D", "") : null;
        if (linhaDig == null)
        {
            var lblLine = LineWith(text, "linha digit");
            if (lblLine != null)
            {
                var digits = Regex.Replace(lblLine, @"\D", "");
                if (digits.Length is >= 47 and <= 48) linhaDig = digits;
            }
        }
        if (linhaDig != null)
        {
            AddF(f, "linha_digitavel", linhaDig, 0.92);
            try
            {
                var info = BrasilApiService.ParseBoleto(linhaDig);
                if (info != null)
                {
                    if (info.Value > 0) AddF(f, "valor", "R$ " + info.Value.ToString("N2", PtBr), 0.90);
                    if (info.DueDate.HasValue) AddF(f, "vencimento", info.DueDate.Value.ToString("dd/MM/yyyy"), 0.88);
                    if (info.BankCode > 0) AddF(f, "banco", info.BankCode.ToString("D3"), 0.80);
                }
            }
            catch { /* linha não-canônica: segue com extração por rótulo */ }
        }

        // Por rótulo (sobrepõe / complementa o parse da linha).
        var valor = Labeled(text, new[] { "valor do documento", "valor do boleto", "valor cobrado", "valor" }, MoneyRx)
                    ?? LastMoney(text);
        if (valor != null) SetF(f, "valor", NormMoney(valor), 0.90);

        var venc = Labeled(text, new[] { "vencimento", "data de vencimento" }, DateRx);
        if (venc != null) SetF(f, "vencimento", venc, 0.90);

        var benef = LabeledText(text, new[] { "beneficiário", "beneficiario", "cedente" });
        if (benef != null) AddF(f, "beneficiario", benef, 0.82);

        var cnpjBenef = NearLabelCnpj(text, new[] { "beneficiário", "beneficiario", "cedente", "cnpj do beneficiário" })
                        ?? FirstValidCnpj(text);
        if (cnpjBenef != null) AddF(f, "cnpj_beneficiario", cnpjBenef, 0.85);

        var banco = LabeledText(text, new[] { "banco" });
        if (banco != null) SetF(f, "banco", banco, 0.78);

        return f;
    }

    private static Dictionary<string, object> ExtractNotaFiscal(string text)
    {
        var f = new Dictionary<string, object>();

        var cnpjs = ValidCnpjs(text);
        if (cnpjs.Count > 0) AddF(f, "cnpj_emitente", cnpjs[0], 0.88);
        if (cnpjs.Count > 1) AddF(f, "cnpj_destinatario", cnpjs[1], 0.82);

        var emit = LabeledText(text, new[] { "razão social", "razao social", "emitente", "nome/razão social", "remetente" });
        if (emit != null) AddF(f, "nome_emitente", emit, 0.78);

        var key = AccessKeyRx.Match(text);
        if (key.Success) AddF(f, "chave_acesso", key.Value, 0.93);

        var emissao = Labeled(text, new[] { "data de emissão", "data emissão", "emissão", "data de emissao" }, DateRx)
                      ?? FirstDate(text);
        if (emissao != null) AddF(f, "data_emissao", emissao, 0.78);

        var total = Labeled(text, new[] { "valor total da nota", "valor total dos produtos", "valor total", "total da nota", "valor da nota" }, MoneyRx)
                    ?? LastMoney(text);
        if (total != null) AddF(f, "valor_total", NormMoney(total), 0.78);

        var numero = Labeled(text, new[] { "nº", "no.", "número", "numero", "nota fiscal nº", "nf-e nº" }, new Regex(@"(\d{1,3}(?:\.\d{3})+|\d{4,9})"));
        if (numero != null) AddF(f, "numero_nota", numero, 0.70);

        var natureza = LabeledText(text, new[] { "natureza da operação", "natureza da operacao", "natureza" });
        if (natureza != null) AddF(f, "natureza_operacao", natureza, 0.70);

        return f;
    }

    private static Dictionary<string, object> ExtractDanfe(string text)
    {
        var f = ExtractNotaFiscal(text);
        var protocolo = Labeled(text, new[] { "protocolo de autorização", "protocolo", "protocolo de autorizacao" }, new Regex(@"(\d{15,16})"));
        if (protocolo != null) AddF(f, "protocolo_autorizacao", protocolo, 0.85);
        var serie = Labeled(text, new[] { "série", "serie" }, new Regex(@"(\d{1,3})"));
        if (serie != null) AddF(f, "serie", serie, 0.72);
        return f;
    }

    private static Dictionary<string, object> ExtractHolerite(string text)
    {
        var f = new Dictionary<string, object>();

        var cnpj = NearLabelCnpj(text, new[] { "empregador", "empresa", "cnpj" }) ?? FirstValidCnpj(text);
        if (cnpj != null) AddF(f, "cnpj_empregador", cnpj, 0.85);

        var empregador = LabeledText(text, new[] { "empregador", "empresa", "razão social" });
        if (empregador != null) AddF(f, "empregador", empregador, 0.75);

        var empregado = LabeledText(text, new[] { "empregado", "funcionário", "funcionario", "nome do empregado", "colaborador" });
        if (empregado != null) AddF(f, "empregado", empregado, 0.75);

        var cpf = Labeled(text, new[] { "cpf do empregado", "cpf" }, CpfRx);
        if (cpf != null) AddF(f, "cpf_empregado", cpf, 0.80);

        var comp = Labeled(text, new[] { "competência", "competencia", "referência", "mês/ano", "mes/ano" }, CompetenciaRx) ?? CompetenciaRx.Match(text).Groups[1].Value;
        if (!string.IsNullOrEmpty(comp)) AddF(f, "competencia", comp, 0.82);

        var liquido = Labeled(text, new[] { "líquido a receber", "liquido a receber", "valor líquido", "salário líquido", "salario liquido", "líquido", "liquido" }, MoneyRx);
        if (liquido != null) AddF(f, "salario_liquido", NormMoney(liquido), 0.80);
        var proventos = Labeled(text, new[] { "total de proventos", "total proventos", "proventos" }, MoneyRx);
        if (proventos != null) AddF(f, "total_proventos", NormMoney(proventos), 0.75);
        var descontos = Labeled(text, new[] { "total de descontos", "total descontos", "descontos" }, MoneyRx);
        if (descontos != null) AddF(f, "total_descontos", NormMoney(descontos), 0.75);
        var baseSal = Labeled(text, new[] { "salário base", "salario base", "salário-base" }, MoneyRx);
        if (baseSal != null) AddF(f, "salario_base", NormMoney(baseSal), 0.72);

        return f;
    }

    private static Dictionary<string, object> ExtractExtrato(string text)
    {
        var f = new Dictionary<string, object>();

        var banco = LabeledText(text, new[] { "banco", "instituição", "instituicao" });
        if (banco != null) AddF(f, "banco", banco, 0.72);

        var ag = Labeled(text, new[] { "agência", "agencia" }, new Regex(@"(\d{3,5}(?:-\d)?)"));
        if (ag != null) AddF(f, "agencia", ag, 0.80);

        var conta = Labeled(text, new[] { "conta corrente", "conta" }, new Regex(@"(\d{4,12}-?\d?)"));
        if (conta != null) AddF(f, "conta", conta, 0.80);

        var titular = LabeledText(text, new[] { "titular", "cliente", "correntista" });
        if (titular != null) AddF(f, "titular", titular, 0.70);

        var dates = DateRx.Matches(text).Select(m => m.Value).Distinct().ToList();
        if (dates.Count >= 1) AddF(f, "periodo_inicio", dates[0], 0.72);
        if (dates.Count >= 2) AddF(f, "periodo_fim", dates[^1], 0.72);

        var sIni = Labeled(text, new[] { "saldo inicial", "saldo anterior" }, MoneyRx);
        if (sIni != null) AddF(f, "saldo_inicial", NormMoney(sIni), 0.72);
        var sFim = Labeled(text, new[] { "saldo final", "saldo atual", "saldo disponível", "saldo" }, MoneyRx);
        if (sFim != null) AddF(f, "saldo_final", NormMoney(sFim), 0.72);

        return f;
    }

    private static Dictionary<string, object> ExtractDarf(string text)
    {
        var f = new Dictionary<string, object>();

        var codigo = Labeled(text, new[] { "código da receita", "codigo da receita", "código", "codigo", "receita" }, new Regex(@"\b(\d{4})\b"));
        if (codigo != null) AddF(f, "codigo_receita", codigo, 0.88);

        var cnpj = FirstValidCnpj(text);
        if (cnpj != null) AddF(f, "cnpj_contribuinte", cnpj, 0.85);

        var periodo = Labeled(text, new[] { "período de apuração", "periodo de apuracao", "apuração", "apuracao" }, new Regex(@"(\d{2}/\d{2}/\d{4}|\d{2}/\d{4})"));
        if (periodo != null) AddF(f, "periodo_apuracao", periodo, 0.78);

        var venc = Labeled(text, new[] { "data de vencimento", "vencimento" }, DateRx) ?? FirstDate(text);
        if (venc != null) AddF(f, "data_vencimento", venc, 0.78);

        var principal = Labeled(text, new[] { "valor principal", "principal" }, MoneyRx);
        if (principal != null) AddF(f, "valor_principal", NormMoney(principal), 0.78);
        var multa = Labeled(text, new[] { "multa" }, MoneyRx);
        if (multa != null) AddF(f, "valor_multa", NormMoney(multa), 0.72);
        var juros = Labeled(text, new[] { "juros" }, MoneyRx);
        if (juros != null) AddF(f, "valor_juros", NormMoney(juros), 0.72);
        var total = Labeled(text, new[] { "valor total", "total a recolher", "total" }, MoneyRx) ?? LastMoney(text);
        if (total != null) AddF(f, "valor_total", NormMoney(total), 0.80);

        return f;
    }

    private static Dictionary<string, object> ExtractGeneric(string text)
    {
        var f = new Dictionary<string, object>();
        var cnpj = FirstValidCnpj(text);
        if (cnpj != null) AddF(f, "cnpj", cnpj, 0.80);
        var money = LastMoney(text);
        if (money != null) AddF(f, "valor", NormMoney(money), 0.65);
        var date = FirstDate(text);
        if (date != null) AddF(f, "data", date, 0.65);
        var key = AccessKeyRx.Match(text);
        if (key.Success) AddF(f, "chave_acesso", key.Value, 0.85);
        return f;
    }

    // ── Helpers de rótulo/padrão ──────────────────────────────────────────

    /// <summary>Linha que contém o rótulo (até a quebra de linha).</summary>
    private static string? LineWith(string text, string label)
    {
        var m = Regex.Match(text, Regex.Escape(label) + @"[^\r\n]*", O);
        return m.Success ? m.Value : null;
    }

    /// <summary>Primeiro valor (padrão) na linha de um dos rótulos.</summary>
    private static string? Labeled(string text, string[] labels, Regex value)
    {
        foreach (var label in labels)
        {
            var line = LineWith(text, label);
            if (line == null) continue;
            // Remove o próprio rótulo p/ não capturar dígitos dele.
            var after = Regex.Replace(line, "^.*?" + Regex.Escape(label), "", O);
            var vm = value.Match(after);
            if (vm.Success) return (vm.Groups.Count > 1 ? vm.Groups[1].Value : vm.Value).Trim();
        }
        return null;
    }

    /// <summary>Texto livre após "rótulo:" (limpo, parando em CNPJ/CPF/duplo-espaço).</summary>
    private static string? LabeledText(string text, string[] labels)
    {
        foreach (var label in labels)
        {
            var m = Regex.Match(text, Regex.Escape(label) + @"\s*[:\-]\s*([^\r\n]{2,90})", O);
            if (!m.Success) continue;
            var v = m.Groups[1].Value.Trim();
            v = Regex.Split(v, @"\s{2,}|\s+CNPJ\b|\s+CPF\b|\s+C[ÓO]DIGO\b", RegexOptions.IgnoreCase)[0].Trim().TrimEnd('.', ',', ';', '-');
            if (v.Length >= 2 && !Regex.IsMatch(v, @"^[\d\.\,/\-\s]+$")) return v;
        }
        return null;
    }

    private static List<string> ValidCnpjs(string text) =>
        CnpjRx.Matches(text).Select(m => m.Value).Where(v => Try(() => Validators.ValidateCnpj(v))).Distinct().ToList();

    private static string? FirstValidCnpj(string text)
    {
        var valid = ValidCnpjs(text).FirstOrDefault();
        if (valid != null) return valid;
        var m = CnpjRx.Match(text);
        return m.Success ? m.Value : null;
    }

    private static string? NearLabelCnpj(string text, string[] labels)
    {
        foreach (var label in labels)
        {
            var idx = text.IndexOf(label, StringComparison.OrdinalIgnoreCase);
            if (idx < 0) continue;
            var window = text.Substring(idx, Math.Min(120, text.Length - idx));
            var m = CnpjRx.Match(window);
            if (m.Success && Try(() => Validators.ValidateCnpj(m.Value))) return m.Value;
        }
        return null;
    }

    private static string? FirstDate(string text) { var m = DateRx.Match(text); return m.Success ? m.Groups[1].Value : null; }

    private static string? LastMoney(string text)
    {
        var m = MoneyRx.Matches(text);
        if (m.Count > 0) return "R$ " + m[^1].Groups[1].Value;
        var b = MoneyBareRx.Matches(text);
        return b.Count > 0 ? "R$ " + b[^1].Groups[1].Value : null;
    }

    private static string NormMoney(string v)
    {
        var m = MoneyBareRx.Match(v);
        return m.Success ? "R$ " + m.Groups[1].Value : v.Trim();
    }

    private static readonly CultureInfo PtBr = CultureInfo.GetCultureInfo("pt-BR");

    private static bool Try(Func<bool> f) { try { return f(); } catch { return false; } }

    private static void AddF(Dictionary<string, object> d, string key, string value, double confidence)
    {
        if (!d.ContainsKey(key) && !string.IsNullOrWhiteSpace(value))
            d[key] = new { text = value.Trim(), confidence = Math.Round(confidence, 4) };
    }

    private static void SetF(Dictionary<string, object> d, string key, string value, double confidence)
    {
        if (!string.IsNullOrWhiteSpace(value))
            d[key] = new { text = value.Trim(), confidence = Math.Round(confidence, 4) };
    }
}
