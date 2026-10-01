using System.Windows;
using System.Windows.Controls;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Views;

public partial class NewUserDialog : Window
{
    public NewUserDialog()
    {
        InitializeComponent();
        cmbRole.SelectedIndex = 0;
    }

    private void OnSave(object sender, RoutedEventArgs e)
    {
        var fullName = txtFullName.Text.Trim();
        var username = txtUsername.Text.Trim().ToLowerInvariant();
        var password = txtPassword.Password;

        if (string.IsNullOrEmpty(fullName))
        {
            ShowError("Informe o nome completo."); return;
        }
        if (string.IsNullOrEmpty(username) || username.Length < 3)
        {
            ShowError("Usuario deve ter pelo menos 3 caracteres."); return;
        }
        if (string.IsNullOrEmpty(password) || password.Length < 4)
        {
            ShowError("Senha deve ter pelo menos 4 caracteres."); return;
        }

        try
        {
            using var db = new AppDbContext();
            
            if (db.Users.Any(u => u.Username == username))
            {
                ShowError("Nome de usuario ja existe."); return;
            }

            var roleTag = (cmbRole.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "Operador";
            var role = roleTag switch
            {
                "Admin" => UserRole.Admin,
                "Visualizador" => UserRole.Visualizador,
                _ => UserRole.Operador
            };

            var user = new User
            {
                FullName = fullName,
                Username = username,
                PasswordHash = PasswordHasher.Hash(password),
                Role = role,
                IsActive = true
            };

            db.Users.Add(user);
            db.SaveChanges();

            DialogResult = true;
            Close();
        }
        catch (Exception ex)
        {
            ShowError($"Erro ao criar usuário: {ex.Message}");
        }
    }

    private void OnCancel(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }

    private void ShowError(string msg)
    {
        txtError.Text = msg;
        txtError.Visibility = Visibility.Visible;
    }
}
