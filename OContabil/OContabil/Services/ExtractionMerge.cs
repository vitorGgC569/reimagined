using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace OContabil.Services;

/// <summary>
/// Funde a extração de um motor ML (GLiNER2) com a determinística (regex/validada),
/// pegando o melhor de cada: nomes/entidades do ML + valor/CNPJ/chave/datas validados do
/// determinístico. Campos "hard" (valor, CNPJ, CPF, chave, datas, linha digitável) vêm do
/// determinístico quando ele os extrai (são validados/padrão-forte); o restante é
/// preenchido pelo ML. Nunca lança — se algo falhar, devolve o resultado ML intacto.
/// </summary>
public static class ExtractionMerge
{
    private static readonly HashSet<string> Hard = new(StringComparer.OrdinalIgnoreCase)
    {
        "valor", "valor_total", "valor_total_nota", "valor_principal", "valor_liquido",
        "salario_liquido", "saldo_final", "saldo_inicial",
        "cnpj", "cnpj_emitente", "cnpj_destinatario", "cnpj_beneficiario", "cnpj_empregador",
        "cnpj_contribuinte", "cpf", "cpf_empregado",
        "chave_acesso", "data", "data_emissao", "data_vencimento", "vencimento",
        "linha_digitavel", "competencia", "periodo_apuracao", "codigo_receita"
    };

    public static GlinerResult Merge(GlinerResult ml, GlinerResult det)
    {
        try
        {
            var mlFields = Flatten(ml.Extraction, out _);
            var detFields = Flatten(det.Extraction, out var detGroup);
            if (detFields.Count == 0) return ml;            // nada a fundir

            // Base: ML (forte em nomes/entidades).
            var merged = new Dictionary<string, JsonElement>(mlFields, StringComparer.OrdinalIgnoreCase);
            // Determinístico: sobrepõe nos campos HARD (validados) e preenche faltantes.
            foreach (var (k, v) in detFields)
                if (Hard.Contains(k) || !merged.ContainsKey(k))
                    merged[k] = v;

            var group = string.IsNullOrEmpty(detGroup) ? "extracao" : detGroup;
            var json = JsonSerializer.Serialize(
                new Dictionary<string, Dictionary<string, JsonElement>> { [group] = merged });
            using var doc = JsonDocument.Parse(json);

            var confs = merged.Values.Select(GetConf).Where(c => c > 0).ToList();
            return new GlinerResult
            {
                Success = true,
                Model = $"{ml.Model}+regex(hibrido)",
                Extraction = doc.RootElement.Clone(),
                EntityCount = merged.Count,
                AvgConfidence = confs.Count > 0 ? Math.Round(confs.Average(), 4) : ml.AvgConfidence,
                ThresholdUsed = ml.ThresholdUsed ?? det.ThresholdUsed,
                OcrText = ml.OcrText ?? det.OcrText,
                TextLength = ml.TextLength,
                Note = "Hibrido: nomes do GLiNER + valor/CNPJ/chave validados do deterministico."
            };
        }
        catch (Exception ex)
        {
            SafeLog.Error("extraction.merge", ex);
            return ml; // qualquer falha: mantém o resultado ML
        }
    }

    /// <summary>Achata {grupo:{campo:{text,confidence}}} → {campo:{text,confidence}} (1º grupo em 'group').</summary>
    private static Dictionary<string, JsonElement> Flatten(JsonElement? ext, out string group)
    {
        group = "";
        var outMap = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        if (ext is not JsonElement e || e.ValueKind != JsonValueKind.Object) return outMap;
        foreach (var g in e.EnumerateObject())
        {
            if (g.Value.ValueKind != JsonValueKind.Object) continue;
            if (string.IsNullOrEmpty(group)) group = g.Name;
            foreach (var f in g.Value.EnumerateObject())
                if (f.Value.ValueKind == JsonValueKind.Object && f.Value.TryGetProperty("text", out _)
                    && !outMap.ContainsKey(f.Name))
                    outMap[f.Name] = f.Value.Clone();
        }
        return outMap;
    }

    private static double GetConf(JsonElement field)
        => field.ValueKind == JsonValueKind.Object
           && field.TryGetProperty("confidence", out var c)
           && c.ValueKind == JsonValueKind.Number ? c.GetDouble() : 0;
}
