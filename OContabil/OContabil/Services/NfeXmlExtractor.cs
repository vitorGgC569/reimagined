using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml;
using System.Xml.Linq;

namespace OContabil.Services;

/// <summary>
/// Extrator ESTRUTURADO de NF-e (XML) — determinístico e mais preciso que regex no
/// texto cru. O parser é XXE-safe: DTD proibido (bloqueia billion-laughs) e
/// XmlResolver nulo (bloqueia entidades externas / leitura de arquivos locais).
/// Robusto a namespace (busca por nome local). Qualquer falha → devolve null e o
/// pipeline segue para o extrator de texto/regex. Nunca lança.
/// </summary>
public static class NfeXmlExtractor
{
    public static bool LooksLikeNfe(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        var head = text.Length > 4000 ? text[..4000] : text;
        if (!head.Contains('<')) return false;
        return head.Contains("infNFe", StringComparison.OrdinalIgnoreCase)
            || head.Contains("portalfiscal.inf.br/nfe", StringComparison.OrdinalIgnoreCase)
            || head.Contains("<NFe", StringComparison.OrdinalIgnoreCase)
            || head.Contains("<nfeProc", StringComparison.OrdinalIgnoreCase);
    }

    public static GlinerResult? TryExtract(string text, double threshold)
    {
        if (!LooksLikeNfe(text)) return null;
        try
        {
            var settings = new XmlReaderSettings
            {
                DtdProcessing = DtdProcessing.Prohibit, // bloqueia DTD / billion-laughs
                XmlResolver = null,                      // bloqueia entidades externas (XXE)
                MaxCharactersFromEntities = 1024,
                IgnoreComments = true,
                IgnoreProcessingInstructions = true,
                CloseInput = true
            };
            using var sr = new StringReader(text);
            using var reader = XmlReader.Create(sr, settings);
            var xdoc = XDocument.Load(reader);

            string? Val(string name) => xdoc.Descendants().FirstOrDefault(e => e.Name.LocalName == name)?.Value?.Trim();
            XElement? El(string name) => xdoc.Descendants().FirstOrDefault(e => e.Name.LocalName == name);
            string? DocIn(XElement? parent) => parent?.Descendants()
                .FirstOrDefault(e => e.Name.LocalName is "CNPJ" or "CPF")?.Value;
            string? NameIn(XElement? parent) => parent?.Descendants()
                .FirstOrDefault(e => e.Name.LocalName == "xNome")?.Value;

            var f = new Dictionary<string, object>();

            // Chave de acesso: infNFe @Id ("NFe"+44 díg) ou <chNFe>.
            var infNFe = El("infNFe");
            var chave = Regex.Replace(infNFe?.Attribute("Id")?.Value ?? "", @"\D", "");
            if (chave.Length != 44) chave = Regex.Replace(Val("chNFe") ?? "", @"\D", "");
            if (chave.Length == 44) Add(f, "chave_acesso", chave, 0.99);

            // Emitente / destinatário.
            var emit = El("emit");
            var docEmit = DocIn(emit);
            if (docEmit != null) Add(f, "cnpj_emitente", FormatDoc(docEmit), 0.99);
            var nomeEmit = NameIn(emit);
            if (nomeEmit != null) Add(f, "nome_emitente", nomeEmit, 0.98);

            var dest = El("dest");
            var docDest = DocIn(dest);
            if (docDest != null) Add(f, "cnpj_destinatario", FormatDoc(docDest), 0.99);
            var nomeDest = NameIn(dest);
            if (nomeDest != null) Add(f, "nome_destinatario", nomeDest, 0.98);

            // Identificação.
            var nNF = Val("nNF");
            if (nNF != null) Add(f, "numero_nota", nNF, 0.98);
            var serie = Val("serie");
            if (serie != null) Add(f, "serie", serie, 0.95);
            var natOp = Val("natOp");
            if (natOp != null) Add(f, "natureza_operacao", natOp, 0.95);
            var data = NormDate(Val("dhEmi") ?? Val("dEmi"));
            if (data != null) Add(f, "data_emissao", data, 0.98);

            // Valor total da nota (ICMSTot/vNF).
            var valor = NormVNF(Val("vNF"));
            if (valor != null) Add(f, "valor_total", valor, 0.99);

            if (f.Count == 0) return null; // não era NF-e útil → deixa o regex tentar

            var wrapped = new Dictionary<string, Dictionary<string, object>> { ["nota_fiscal"] = f };
            var json = JsonSerializer.Serialize(wrapped);
            using var jdoc = JsonDocument.Parse(json);

            return new GlinerResult
            {
                Success = true,
                Model = "nfe-xml",
                Extraction = jdoc.RootElement.Clone(),
                AvgConfidence = 0.97,
                EntityCount = f.Count,
                ThresholdUsed = threshold,
                OcrText = text.Length > 1500 ? text[..1500] + "..." : text,
                TextLength = text.Length,
                Note = "Extracao estruturada de NF-e (XML, parser XXE-safe)."
            };
        }
        catch (Exception ex)
        {
            SafeLog.Error("nfexml", ex);
            return null; // qualquer falha → cai para o extrator de texto/regex
        }
    }

    private static string FormatDoc(string raw)
    {
        var digits = Regex.Replace(raw, @"\D", "");
        if (digits.Length == 14 && ulong.TryParse(digits, out var c))
            return c.ToString(@"00\.000\.000\/0000\-00", CultureInfo.InvariantCulture);
        if (digits.Length == 11 && ulong.TryParse(digits, out var p))
            return p.ToString(@"000\.000\.000\-00", CultureInfo.InvariantCulture);
        return digits;
    }

    private static string? NormVNF(string? v)
    {
        if (string.IsNullOrWhiteSpace(v)) return null;
        return decimal.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out var d)
            ? "R$ " + d.ToString("N2", CultureInfo.GetCultureInfo("pt-BR"))
            : null;
    }

    private static string? NormDate(string? v)
    {
        if (string.IsNullOrWhiteSpace(v)) return null;
        return DateTimeOffset.TryParse(v, CultureInfo.InvariantCulture, DateTimeStyles.None, out var dto)
            ? dto.ToString("dd/MM/yyyy")
            : null;
    }

    private static void Add(Dictionary<string, object> d, string key, string value, double conf)
    {
        if (!d.ContainsKey(key) && !string.IsNullOrWhiteSpace(value))
            d[key] = new { text = value.Trim(), confidence = Math.Round(conf, 4) };
    }
}
