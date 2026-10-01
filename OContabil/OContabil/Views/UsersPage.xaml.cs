using System.Windows;
using System.Windows.Controls;
using OContabil.Data;
using OContabil.Services;

namespace OContabil.Views;

public partial class UsersPage : UserControl
{
    private readonly AuthService _auth;

    public UsersPage(AuthService auth)
    {
        InitializeComponent();
        _auth = auth;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        Refresh();
    }

    private void Refresh()
    {
        try
        {
            using var db = new AppDbContext();
            var users = db.Users.OrderBy(u => u.FullName).ToList();
            gridUsers.ItemsSource = users.Select(u => new
            {
                u.FullName,
                u.Username,
                u.RoleDisplay,
                StatusStr = u.IsActive ? "Ativo" : "Inativo",
                CreatedStr = u.CreatedAt.ToString("dd/MM/yyyy")
            }).ToList();
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao carregar usuários: {ex.Message}");
        }
    }

    private void OnAdd(object sender, RoutedEventArgs e)
    {
        if (!_auth.CanManageUsers)
        {
            MessageBox.Show("Apenas administradores podem criar usuarios.", "Acesso negado",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var dialog = new NewUserDialog() { Owner = Window.GetWindow(this) };
        if (dialog.ShowDialog() == true)
        {
            Refresh();
        }
    }
}
