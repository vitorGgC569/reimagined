using System.ComponentModel;
using System.Windows;
using ContaDocAI.ViewModels;
using ContaDocAI.Views;

namespace ContaDocAI;

public partial class MainWindow : Window
{
    private readonly MainViewModel _vm;

    public MainWindow()
    {
        InitializeComponent();
        _vm = (MainViewModel)DataContext;
        _vm.PropertyChanged += OnViewModelPropertyChanged;

        // Initial page
        NavigateToPage("Dashboard");
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(MainViewModel.CurrentPage))
        {
            NavigateToPage(_vm.CurrentPage);
        }
    }

    private void NavigateToPage(string page)
    {
        pageHost.Content = page switch
        {
            "Dashboard" => new DashboardView(),
            "Upload" => new UploadView(),
            "Validation" => new ValidationView(),
            "Clients" => new ClientsView(),
            "Settings" => new SettingsView(),
            _ => new DashboardView()
        };
    }
}