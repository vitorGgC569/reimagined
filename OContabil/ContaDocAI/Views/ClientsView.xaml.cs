using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using ContaDocAI.Services;

namespace ContaDocAI.Views;

public partial class ClientsView : UserControl
{
    public ClientsView()
    {
        InitializeComponent();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        var clients = MockDataService.Clients;
        txtClientCount.Text = clients.Count.ToString();
        txtTotalDocs.Text = clients.Sum(c => c.DocsMonth).ToString("N0");
        txtPendingTotal.Text = clients.Sum(c => c.PendingDocs).ToString();
        PopulateCards(clients.Select(ToViewModel).ToList());
    }

    private object ToViewModel(Models.Client c)
    {
        var successBrush = (Brush)FindResource("SuccessBrush");
        var warningBrush = (Brush)FindResource("WarningBrush");
        var successBg = (Brush)FindResource("SuccessBgBrush");
        var warningBg = (Brush)FindResource("WarningBgBrush");
        var accentBrush = (Brush)FindResource("AccentBrush");
        var cardWidth = 280.0; // approximate usable width inside card

        return new
        {
            c.Name, c.Cnpj, c.Initials, c.Color,
            c.DocsMonth, c.DocsValidated, c.PendingDocs,
            c.ValidationPercent,
            StatusText = c.IsUpToDate ? "Em dia" : $"{c.PendingDocs} pend.",
            BadgeBg = c.IsUpToDate ? successBg : warningBg,
            BadgeFg = c.IsUpToDate ? successBrush : warningBrush,
            ProgressWidth = c.ValidationPercent / 100.0 * cardWidth,
            ProgressColor = c.ValidationPercent == 100 ? successBrush : accentBrush,
            PendingColor = c.PendingDocs > 0 ? warningBrush : successBrush,
        };
    }

    private void PopulateCards(List<object> items)
    {
        clientsPanel.ItemsSource = items;
    }

    private void OnSearch(object sender, TextChangedEventArgs e)
    {
        var q = searchBox.Text.ToLowerInvariant();
        var filtered = MockDataService.Clients
            .Where(c => c.Name.Contains(q, StringComparison.OrdinalIgnoreCase) || c.Cnpj.Contains(q))
            .Select(ToViewModel)
            .ToList();
        PopulateCards(filtered);
    }

    private void OnAddClient(object sender, RoutedEventArgs e)
    {
        MessageBox.Show("Formulario de cadastro de novo cliente sera implementado na proxima versao.",
            "ContaDoc AI — Novo Cliente", MessageBoxButton.OK, MessageBoxImage.Information);
    }
}
