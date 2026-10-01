using System.ComponentModel.DataAnnotations;

namespace OContabil.Models;

public class Client
{
    public int Id { get; set; }

    [Required, MaxLength(200)]
    public string Name { get; set; } = "";

    [Required, MaxLength(20)]
    public string Cnpj { get; set; } = "";

    [MaxLength(200)]
    public string Email { get; set; } = "";

    [MaxLength(20)]
    public string Phone { get; set; } = "";

    [MaxLength(50)]
    public string TaxRegime { get; set; } = "Simples Nacional";

    public bool IsActive { get; set; } = true;

    public DateTime CreatedAt { get; set; } = DateTime.Now;

    public int DocumentCount { get; set; }
    public int ValidatedCount { get; set; }
    public int PendingCount => DocumentCount - ValidatedCount;

    public string Initials => string.Join("",
        Name.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Where(w => w.Length > 0)
            .Take(2)
            .Select(w => char.ToUpper(w[0])));
}
