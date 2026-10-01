using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OContabil.Services;

/// <summary>
/// Parser de extratos bancários no formato OFX (Open Financial Exchange) 1.x
/// (SGML) e 2.x (XML). Retorna lista de transações com data, descrição, valor
/// e tipo (CRÉDITO/DÉBITO). Cobertura suficiente para os principais bancos
/// brasileiros (Itaú, Bradesco, Santander, Caixa).
/// </summary>
public sealed class OfxParserService
{
    public sealed record OfxTransaction(
        DateTime Date,
        string FitId,
        string Type,
        decimal Amount,
        string Memo,
        string? CheckNum);

    public sealed record OfxStatement(
        string? BankId,
        string? AccountId,
        string? AccountType,
        string Currency,
        DateTime? StartDate,
        DateTime? EndDate,
        decimal? LedgerBalance,
        DateTime? BalanceDate,
        List<OfxTransaction> Transactions);

    public OfxStatement Parse(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("Arquivo OFX não encontrado.", path);

        // OFX 1.x usa Latin1 por padrão (CHARSET:1252). 2.x é XML UTF-8.
        var bytes = File.ReadAllBytes(path);
        var content = TryRead(bytes, Encoding.UTF8) ?? TryRead(bytes, Encoding.Latin1) ?? "";
        if (string.IsNullOrWhiteSpace(content))
            throw new InvalidDataException("Arquivo OFX vazio ou ilegível.");

        // Localiza início do <OFX>
        var idx = content.IndexOf("<OFX>", StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            throw new InvalidDataException("Arquivo não contém bloco <OFX>.");
        var body = content.Substring(idx);

        return new OfxStatement(
            BankId: ReadValue(body, "BANKID"),
            AccountId: ReadValue(body, "ACCTID"),
            AccountType: ReadValue(body, "ACCTTYPE"),
            Currency: ReadValue(body, "CURDEF") ?? "BRL",
            StartDate: ParseOfxDate(ReadValue(body, "DTSTART")),
            EndDate: ParseOfxDate(ReadValue(body, "DTEND")),
            LedgerBalance: ParseAmount(ReadValue(body, "BALAMT")),
            BalanceDate: ParseOfxDate(ReadValue(body, "DTASOF")),
            Transactions: ParseTransactions(body));
    }

    public string ToJson(OfxStatement stmt) =>
        JsonSerializer.Serialize(stmt, new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });

    private static string? TryRead(byte[] bytes, Encoding encoding)
    {
        try { return encoding.GetString(bytes); } catch { return null; }
    }

    private static List<OfxTransaction> ParseTransactions(string body)
    {
        var list = new List<OfxTransaction>();
        var rx = new Regex(@"<STMTTRN>(?<inner>.*?)</STMTTRN>",
            RegexOptions.Singleline | RegexOptions.IgnoreCase);

        foreach (Match m in rx.Matches(body))
        {
            var inner = m.Groups["inner"].Value;
            var date = ParseOfxDate(ReadValue(inner, "DTPOSTED")) ?? DateTime.MinValue;
            var fitId = ReadValue(inner, "FITID") ?? "";
            var type = ReadValue(inner, "TRNTYPE") ?? "OUTRO";
            var amount = ParseAmount(ReadValue(inner, "TRNAMT")) ?? 0m;
            var memo = ReadValue(inner, "MEMO") ?? ReadValue(inner, "NAME") ?? "";
            var check = ReadValue(inner, "CHECKNUM");
            list.Add(new OfxTransaction(date, fitId, type, amount, memo, check));
        }
        return list;
    }

    private static string? ReadValue(string body, string tag)
    {
        // OFX 1.x: <TAG>valor (sem fechamento) ou <TAG>valor</TAG>
        // OFX 2.x: <TAG>valor</TAG>
        var rx = new Regex($@"<{tag}>(?<v>[^<\r\n]+?)(?:</{tag}>|\r|\n|$)",
            RegexOptions.IgnoreCase);
        var m = rx.Match(body);
        return m.Success ? m.Groups["v"].Value.Trim() : null;
    }

    private static DateTime? ParseOfxDate(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        // Formato OFX: YYYYMMDD[HHMMSS][.XXX][ ±HHMM]
        var digits = new string(raw.Where(char.IsDigit).ToArray());
        if (digits.Length < 8) return null;
        try
        {
            int y = int.Parse(digits.Substring(0, 4));
            int M = int.Parse(digits.Substring(4, 2));
            int d = int.Parse(digits.Substring(6, 2));
            if (digits.Length >= 14)
            {
                int h = int.Parse(digits.Substring(8, 2));
                int min = int.Parse(digits.Substring(10, 2));
                int sec = int.Parse(digits.Substring(12, 2));
                return new DateTime(y, M, d, h, min, sec);
            }
            return new DateTime(y, M, d);
        }
        catch { return null; }
    }

    private static decimal? ParseAmount(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        var v = raw.Replace(',', '.');
        if (decimal.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var x))
            return x;
        return null;
    }
}
