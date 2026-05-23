using System.ComponentModel.DataAnnotations;

namespace OContabil.Models;

public enum UserRole
{
    Admin,
    Operador,
    Visualizador
}

public class User
{
    public int Id { get; set; }

    [Required, MaxLength(50)]
    public string Username { get; set; } = "";

    [Required]
    public string PasswordHash { get; set; } = "";

    [Required, MaxLength(120)]
    public string FullName { get; set; } = "";

    public UserRole Role { get; set; } = UserRole.Operador;

    public bool IsActive { get; set; } = true;

    public DateTime CreatedAt { get; set; } = DateTime.Now;

    public DateTime? LastLoginAt { get; set; }

    [MaxLength(200)]
    public string? Email { get; set; }

    public bool MustChangePassword { get; set; }

    public string RoleDisplay => Role switch
    {
        UserRole.Admin => "Administrador",
        UserRole.Operador => "Operador",
        UserRole.Visualizador => "Visualizador",
        _ => "Desconhecido"
    };
}
