using System.ComponentModel.DataAnnotations;

namespace OContabil.Models;

public class ChartOfAccount
{
    public int Id { get; set; }

    [Required, StringLength(20)]
    public string Code { get; set; } = string.Empty;

    [Required, StringLength(100)]
    public string Description { get; set; } = string.Empty;

    [Required]
    public string Type { get; set; } = "Despesa"; // Ativo, Passivo, Receita, Despesa

    // Optional Regex pattern to auto-classify OCR text (e.g., "ENEL|CEMIG|ELEKTRO")
    public string? AutoClassificationRegex { get; set; }
}
