namespace OContabil.Services;

/// <summary>
/// Escrita CSV segura. Previne CSV/Formula Injection (CWE-1236): células que
/// começam com '=', '+', '-', '@' ou tab são prefixadas com apóstrofo para que
/// Excel/LibreOffice não as executem como fórmula. Também faz o quoting estilo
/// RFC-4180 (aspas duplicadas, e células com separador/quebra entre aspas).
/// Separador padrão ';' (Excel pt-BR).
/// </summary>
public static class SecureCsv
{
    public static string Cell(string? s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        if (s[0] is '=' or '+' or '-' or '@' or '\t')
            s = "'" + s;
        var needsQuote = s.Contains(';') || s.Contains('"') || s.Contains('\n') || s.Contains('\r');
        s = s.Replace("\"", "\"\"").Replace("\r", " ").Replace("\n", " ");
        return needsQuote ? "\"" + s + "\"" : s;
    }
}
