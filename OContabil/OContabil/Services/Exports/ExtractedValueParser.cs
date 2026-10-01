using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OContabil.Services.Exports;

/// <summary>
/// Lê valores e datas a partir do JSON extraído pela IA. Suporta tanto o
/// formato simples ("campo": "valor") quanto o aninhado ("campo": { "text": "valor" }).
/// </summary>
public static class ExtractedValueParser
{
    public static decimal ParseMoney(string? extractedJson, params string[] fieldCandidates)
    {
        var raw = ReadField(extractedJson, fieldCandidates);
        if (string.IsNullOrWhiteSpace(raw)) return 0m;

        var digits = Regex.Replace(raw, @"[^\d,\.\-]", "");
        if (string.IsNullOrEmpty(digits)) return 0m;

        // Suporta tanto formato BR (1.234,56) quanto US (1234.56)
        var normalized = digits;
        if (normalized.Contains(',') && normalized.Contains('.'))
        {
            if (normalized.LastIndexOf(',') > normalized.LastIndexOf('.'))
                normalized = normalized.Replace(".", "").Replace(",", ".");
            else
                normalized = normalized.Replace(",", "");
        }
        else if (normalized.Contains(','))
        {
            normalized = normalized.Replace(",", ".");
        }

        return decimal.TryParse(normalized, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0m;
    }

    public static DateTime? ParseDate(string? extractedJson, params string[] fieldCandidates)
    {
        var raw = ReadField(extractedJson, fieldCandidates);
        if (string.IsNullOrWhiteSpace(raw)) return null;

        var match = Regex.Match(raw, @"(\d{2})/(\d{2})/(\d{4})");
        if (match.Success && DateTime.TryParseExact(match.Value, "dd/MM/yyyy",
            CultureInfo.InvariantCulture, DateTimeStyles.None, out var dt))
            return dt;

        if (DateTime.TryParse(raw, CultureInfo.GetCultureInfo("pt-BR"), DateTimeStyles.None, out var dt2))
            return dt2;

        return null;
    }

    public static string ReadField(string? extractedJson, params string[] candidates)
    {
        if (string.IsNullOrWhiteSpace(extractedJson)) return "";

        try
        {
            using var doc = JsonDocument.Parse(extractedJson);
            return FindRecursive(doc.RootElement, candidates) ?? "";
        }
        catch { return ""; }
    }

    private static string? FindRecursive(JsonElement element, string[] candidates)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (var prop in element.EnumerateObject())
            {
                if (candidates.Contains(prop.Name, StringComparer.OrdinalIgnoreCase))
                {
                    return AsText(prop.Value);
                }
                var nested = FindRecursive(prop.Value, candidates);
                if (!string.IsNullOrEmpty(nested)) return nested;
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                var nested = FindRecursive(item, candidates);
                if (!string.IsNullOrEmpty(nested)) return nested;
            }
        }
        return null;
    }

    private static string AsText(JsonElement value)
    {
        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString() ?? "",
            JsonValueKind.Number => value.GetRawText(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            JsonValueKind.Object => value.TryGetProperty("text", out var t) ? AsText(t) : value.GetRawText(),
            JsonValueKind.Array => string.Join(", ", value.EnumerateArray().Select(AsText)),
            _ => ""
        };
    }
}
