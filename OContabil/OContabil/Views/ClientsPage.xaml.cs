using System.Windows;
using System.Windows.Controls;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Views;

public partial class ClientsPage : UserControl
{
    private readonly AuthService _auth;
    private List<Client> _all = new();

    public ClientsPage(AuthService auth)
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
            _all = db.Clients.Where(c => c.IsActive).OrderBy(c => c.Name).ToList();
            gridClients.ItemsSource = _all;
            txtCount.Text = $"{_all.Count} cliente(s)";
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao carregar clientes: {ex.Message}");
        }
    }

    private void OnSearch(object sender, TextChangedEventArgs e)
    {
        var q = txtSearch.Text.Trim();
        if (string.IsNullOrEmpty(q))
        {
            gridClients.ItemsSource = _all;
            txtCount.Text = $"{_all.Count} cliente(s)";
            return;
        }
        var filtered = _all.Where(c =>
            c.Name.Contains(q, StringComparison.OrdinalIgnoreCase) ||
            c.Cnpj.Contains(q)).ToList();
        gridClients.ItemsSource = filtered;
        txtCount.Text = $"{filtered.Count} cliente(s)";
    }

    private void OnAdd(object sender, RoutedEventArgs e)
    {
        if (!_auth.CanEdit)
        {
            ToastService.ShowWarning("Sem permissao para cadastrar clientes.");
            return;
        }

        var dialog = new NewClientDialog() { Owner = Window.GetWindow(this) };
        if (dialog.ShowDialog() == true)
        {
            Refresh();
        }
    }

    private void OnDeleteClient(object sender, RoutedEventArgs e)
    {
        if (!_auth.CanEdit)
        {
            ToastService.ShowWarning("Sem permissão para excluir clientes.");
            return;
        }

        if (sender is FrameworkElement el && el.DataContext is Client client)
        {
            // BUG-09 FIX: Re-fetch from a fresh context by ID instead of using the cached entity
            try
            {
                using var db = new AppDbContext();
                var freshClient = db.Clients.Find(client.Id);
                if (freshClient == null) return;

                if (freshClient.DocumentCount > 0)
                {
                    ToastService.ShowError($"Não é possível excluir o cliente {freshClient.Name} pois ele possui {freshClient.DocumentCount} documento(s) vinculado(s).");
                    return;
                }

                if (MessageBox.Show($"Tem certeza que deseja EXCLUIR o cliente {freshClient.Name}?",
                    "Excluir Cliente", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
                {
                    db.Clients.Remove(freshClient);
                    db.SaveChanges();
                    Refresh();
                }
            }
            catch (Exception ex)
            {
                ToastService.ShowError($"Erro ao excluir cliente: {ex.Message}");
            }
        }
    }
}
