using System.Globalization;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OContabil.Services;

/// <summary>
/// Unified BrasilAPI client for all public data: CNPJ, CEP, Feriados, IBGE, Bancos.
/// All endpoints are free, no auth required.
/// </summary>
public class BrasilApiService
{
    private static readonly HttpClient _http = new()
    {
        Timeout = TimeSpan.FromSeconds(15),
        BaseAddress = new Uri("https://brasilapi.com.br/api/")
    };

    // ═══════════════════════════════════════════════════════════════
    // CNPJ Lookup
    // ═══════════════════════════════════════════════════════════════

    public async Task<CnpjResult?> LookupCnpjAsync(string cnpj)
    {
        var digits = StripNonDigits(cnpj);
        if (digits.Length != 14) return null;

        try
        {
            var json = await _http.GetStringAsync($"cnpj/v1/{digits}");
            return JsonSerializer.Deserialize<CnpjResult>(json, _jsonOpts);
        }
        catch (Exception ex)
        { 
            // Return a fake result with the error string in RazaoSocial so the UI can show it for debugging
            return new CnpjResult { RazaoSocial = $"ERRO_API: {ex.Message}" }; 
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // CEP Lookup
    // ═══════════════════════════════════════════════════════════════

    public async Task<CepResult?> LookupCepAsync(string cep)
    {
        var digits = StripNonDigits(cep);
        if (digits.Length != 8) return null;

        try
        {
            var json = await _http.GetStringAsync($"cep/v2/{digits}");
            return JsonSerializer.Deserialize<CepResult>(json, _jsonOpts);
        }
        catch { return null; }
    }

    // ═══════════════════════════════════════════════════════════════
    // Feriados Nacionais
    // ═══════════════════════════════════════════════════════════════

    public async Task<List<Feriado>> GetFeriadosAsync(int year)
    {
        try
        {
            var json = await _http.GetStringAsync($"feriados/v3/{year}");
            return JsonSerializer.Deserialize<List<Feriado>>(json, _jsonOpts) ?? new();
        }
        catch { return new(); }
    }

    /// <summary>
    /// Calculates the valid payment date considering weekends and national holidays.
    /// </summary>
    public async Task<(DateTime validDate, string reason)> GetValidPaymentDateAsync(DateTime dueDate)
    {
        var feriados = await GetFeriadosAsync(dueDate.Year);
        var feriadoDates = feriados
            .Where(f => DateTime.TryParse(f.Date, out _))
            .Select(f => DateTime.Parse(f.Date).Date)
            .ToHashSet();

        var current = dueDate.Date;
        var originalDate = current;

        while (current.DayOfWeek == DayOfWeek.Saturday ||
               current.DayOfWeek == DayOfWeek.Sunday ||
               feriadoDates.Contains(current))
        {
            current = current.AddDays(1);
        }

        if (current == originalDate)
            return (current, "");

        var reason = originalDate.DayOfWeek switch
        {
            DayOfWeek.Saturday => "Sabado",
            DayOfWeek.Sunday => "Domingo",
            _ => feriadoDates.Contains(originalDate)
                ? feriados.FirstOrDefault(f => DateTime.TryParse(f.Date, out var d) && d.Date == originalDate)?.Name ?? "Feriado"
                : "Feriado"
        };

        return (current, $"Vencimento original: {originalDate:dd/MM} ({reason}). Valido: {current:dd/MM}");
    }

    // ═══════════════════════════════════════════════════════════════
    // IBGE — City Codes
    // ═══════════════════════════════════════════════════════════════

    public async Task<string?> GetIbgeCodeAsync(string cityName, string uf)
    {
        try
        {
            var json = await _http.GetStringAsync($"ibge/municipios/v1/{uf}?providers=dados-abertos-br,gov,wikipedia");
            var cities = JsonSerializer.Deserialize<List<IbgeCity>>(json, _jsonOpts) ?? new();

            var match = cities.FirstOrDefault(c =>
                string.Equals(c.Nome, cityName, StringComparison.OrdinalIgnoreCase));

            return match?.CodigoIbge;
        }
        catch { return null; }
    }

    // ═══════════════════════════════════════════════════════════════
    // Banks (ISPB)
    // ═══════════════════════════════════════════════════════════════

    public async Task<List<Bank>> GetBanksAsync()
    {
        try
        {
            var json = await _http.GetStringAsync("banks/v1");
            return JsonSerializer.Deserialize<List<Bank>>(json, _jsonOpts) ?? new();
        }
        catch { return new(); }
    }

    public async Task<Bank?> GetBankByCodeAsync(int code)
    {
        try
        {
            var json = await _http.GetStringAsync($"banks/v1/{code}");
            return JsonSerializer.Deserialize<Bank>(json, _jsonOpts);
        }
        catch { return null; }
    }

    // ═══════════════════════════════════════════════════════════════
    // Boleto Validation (mathematical line digitável)
    // ═══════════════════════════════════════════════════════════════

    public static BoletoInfo? ParseBoleto(string linhaDigitavel)
    {
        var digits = StripNonDigits(linhaDigitavel);

        // Boleto bancário: 47 dígitos
        if (digits.Length == 47)
        {
            return new BoletoInfo
            {
                BankCode = int.TryParse(digits[..3], out var bc) ? bc : 0,
                Value = decimal.TryParse(digits[37..47], NumberStyles.Any, CultureInfo.InvariantCulture, out var v)
                    ? v / 100m : 0,
                DueFactor = int.TryParse(digits[33..37], out var df) ? df : 0,
                IsValid = ValidateBoletoCheckDigit(digits),
                Type = "Bancario"
            };
        }

        // Boleto concessionária: 48 dígitos
        if (digits.Length == 48)
        {
            return new BoletoInfo
            {
                Value = decimal.TryParse(digits[4..15], NumberStyles.Any, CultureInfo.InvariantCulture, out var v)
                    ? v / 100m : 0,
                IsValid = true,
                Type = "Concessionaria"
            };
        }

        return null;
    }

    private static bool ValidateBoletoCheckDigit(string digits)
    {
        if (digits.Length != 47) return false;

        // Construct barcode from linha digitável
        var barcode = digits[..4] + digits[32..47] + digits[4..9] + digits[10..20] + digits[21..31];
        if (barcode.Length != 44) return false;

        var checkDigit = barcode[4] - '0';
        var withoutCheck = barcode[..4] + barcode[5..];

        int sum = 0;
        int weight = 2;
        for (int i = withoutCheck.Length - 1; i >= 0; i--)
        {
            sum += (withoutCheck[i] - '0') * weight;
            weight = weight == 9 ? 2 : weight + 1;
        }

        var remainder = sum % 11;
        var expected = (remainder == 0 || remainder == 1 || remainder == 10) ? 1 : 11 - remainder;

        return checkDigit == expected;
    }

    // ═══════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════

    private static string StripNonDigits(string input) =>
        System.Text.RegularExpressions.Regex.Replace(input ?? "", @"[^\d]", "");

    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        PropertyNameCaseInsensitive = true
    };
}

// ═══════════════════════════════════════════════════════════════
// DTOs
// ═══════════════════════════════════════════════════════════════

public class CnpjResult
{
    [JsonPropertyName("cnpj")] public string Cnpj { get; set; } = "";
    [JsonPropertyName("razao_social")] public string RazaoSocial { get; set; } = "";
    [JsonPropertyName("nome_fantasia")] public string NomeFantasia { get; set; } = "";
    [JsonPropertyName("descricao_situacao_cadastral")] public string SituacaoCadastral { get; set; } = "";
    [JsonPropertyName("situacao_cadastral")] public int? SituacaoCadastralCodigo { get; set; }
    [JsonPropertyName("logradouro")] public string? Logradouro { get; set; }
    [JsonPropertyName("numero")] public string? Numero { get; set; }
    [JsonPropertyName("complemento")] public string? Complemento { get; set; }
    [JsonPropertyName("bairro")] public string? Bairro { get; set; }
    [JsonPropertyName("municipio")] public string? Municipio { get; set; }
    [JsonPropertyName("uf")] public string? Uf { get; set; }
    [JsonPropertyName("cep")] public string? Cep { get; set; }
    [JsonPropertyName("ddd_telefone_1")] public string? Telefone1 { get; set; }
    [JsonPropertyName("ddd_telefone_2")] public string? Telefone2 { get; set; }
    [JsonPropertyName("email")] public string? Email { get; set; }
    [JsonPropertyName("cnae_fiscal_descricao")] public string? AtividadePrincipal { get; set; }
    [JsonPropertyName("natureza_juridica")] public string? NaturezaJuridica { get; set; }
    
    // Some CNPJs return null or string for CapitalSocial, so let's use JsonElement to be safe
    [JsonPropertyName("capital_social")] public JsonElement? CapitalSocialRaw { get; set; }
    
    public decimal CapitalSocial 
    {
        get
        {
            if (CapitalSocialRaw == null) return 0m;
            if (CapitalSocialRaw.Value.ValueKind == JsonValueKind.Number) return CapitalSocialRaw.Value.GetDecimal();
            if (CapitalSocialRaw.Value.ValueKind == JsonValueKind.String && decimal.TryParse(CapitalSocialRaw.Value.GetString()?.Replace('.', ','), out var v)) return v;
            return 0m;
        }
    }

    [JsonPropertyName("data_inicio_atividade")] public string? DataInicioAtividade { get; set; }
    [JsonPropertyName("porte")] public string? Porte { get; set; }
    [JsonPropertyName("opcao_pelo_simples")] public JsonElement? OpcaoSimplesRaw { get; set; }

    public bool OpcaoSimples
    {
        get
        {
            if (OpcaoSimplesRaw == null) return false;
            if (OpcaoSimplesRaw.Value.ValueKind == JsonValueKind.True) return true;
            if (OpcaoSimplesRaw.Value.ValueKind == JsonValueKind.False) return false;
            return false;
        }
    }

    public string EnderecoCompleto =>
        $"{Logradouro}, {Numero}" +
        (string.IsNullOrWhiteSpace(Complemento) ? "" : $" - {Complemento}") +
        $", {Bairro}, {Municipio}/{Uf}" +
        (string.IsNullOrWhiteSpace(Cep) ? "" : $" - CEP {Cep}");

    public bool IsAtiva => SituacaoCadastralCodigo == 2;

    public string RegimeEstimado =>
        OpcaoSimples ? "Simples Nacional" :
        Porte == "DEMAIS" ? "Lucro Real" : "Lucro Presumido";

    public string TelefoneFormatado
    {
        get
        {
            var digits = System.Text.RegularExpressions.Regex.Replace(Telefone1 ?? "", @"[^\d]", "");
            if (digits.Length == 11) return $"({digits[..2]}) {digits[2..7]}-{digits[7..11]}";
            if (digits.Length == 10) return $"({digits[..2]}) {digits[2..6]}-{digits[6..10]}";
            return Telefone1 ?? "";
        }
    }
}

public class CepResult
{
    [JsonPropertyName("cep")] public string Cep { get; set; } = "";
    [JsonPropertyName("state")] public string State { get; set; } = "";
    [JsonPropertyName("city")] public string City { get; set; } = "";
    [JsonPropertyName("neighborhood")] public string Neighborhood { get; set; } = "";
    [JsonPropertyName("street")] public string Street { get; set; } = "";
}

public class Feriado
{
    [JsonPropertyName("date")] public string Date { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("type")] public string Type { get; set; } = "";
}

public class IbgeCity
{
    [JsonPropertyName("nome")] public string Nome { get; set; } = "";
    [JsonPropertyName("codigo_ibge")] public string CodigoIbge { get; set; } = "";
}

public class Bank
{
    [JsonPropertyName("ispb")] public string Ispb { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("code")] public int? Code { get; set; }
    [JsonPropertyName("fullName")] public string FullName { get; set; } = "";
}

public class BoletoInfo
{
    public int BankCode { get; set; }
    public decimal Value { get; set; }
    public int DueFactor { get; set; }
    public bool IsValid { get; set; }
    public string Type { get; set; } = "";

    public DateTime? DueDate
    {
        get
        {
            if (DueFactor <= 0) return null;
            return new DateTime(1997, 10, 7).AddDays(DueFactor);
        }
    }
}
