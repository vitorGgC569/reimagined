using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using OContabil.Data;
using OContabil.Services;

namespace OContabil.Views;

public partial class MainWindow : Window
{
    private readonly AuthService _auth;
    private string _currentPage = "Dashboard";

    public MainWindow(AuthService auth)
    {
        InitializeComponent();
        _auth = auth;

        txtUserName.Text = auth.CurrentUser!.FullName;
        txtUserRole.Text = auth.CurrentUser.RoleDisplay;
        txtUserInitials.Text = string.Join("",
            auth.CurrentUser.FullName.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Take(2).Select(w => char.ToUpper(w[0])));

        if (!auth.CanManageUsers)
            navUsers.Visibility = Visibility.Collapsed;

        ToastService.Initialize(this);

        // Sessão com timeout configurável (default 20 min).
        _auth.Session.Timeout = TimeSpan.FromMinutes(AppSettings.SessionTimeoutMinutes);
        _auth.Session.Start(auth.CurrentUser, this);
        _auth.Session.SessionExpired += OnSessionExpired;

        BackupService.Instance.ConfigureFromSettings();
        BackupService.Instance.Start();

        // Atalhos de teclado globais.
        InputBindings.Add(new KeyBinding(new RelayCommand(() => Navigate("Dashboard")),
            new KeyGesture(System.Windows.Input.Key.D1, System.Windows.Input.ModifierKeys.Alt)));
        InputBindings.Add(new KeyBinding(new RelayCommand(() => Navigate("Documents")),
            new KeyGesture(System.Windows.Input.Key.D2, System.Windows.Input.ModifierKeys.Alt)));
        InputBindings.Add(new KeyBinding(new RelayCommand(() => Navigate("Clients")),
            new KeyGesture(System.Windows.Input.Key.D3, System.Windows.Input.ModifierKeys.Alt)));
        InputBindings.Add(new KeyBinding(new RelayCommand(() => ThemeManager.ToggleTheme()),
            new KeyGesture(System.Windows.Input.Key.T, System.Windows.Input.ModifierKeys.Control)));
        InputBindings.Add(new KeyBinding(new RelayCommand(() =>
        {
            var dlg = new ShortcutsHelpDialog { Owner = this };
            dlg.ShowDialog();
        }), new KeyGesture(System.Windows.Input.Key.F1)));

        Navigate("Dashboard");

        if (auth.CurrentUser.MustChangePassword)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                MessageBox.Show(
                    "A senha de administrador padrão (admin/admin) precisa ser alterada para garantir a segurança do sistema. Acesse Usuários para definir uma nova senha.",
                    "Segurança", MessageBoxButton.OK, MessageBoxImage.Warning);
            }));
        }
    }

    private void OnSessionExpired()
    {
        Dispatcher.Invoke(() =>
        {
            MessageBox.Show(
                "Sua sessão expirou por inatividade. Faça login novamente para continuar.",
                "Sessão expirada", MessageBoxButton.OK, MessageBoxImage.Information);
            _auth.Logout();
            var login = new LoginView();
            login.Show();
            Close();
        });
    }

    protected override void OnClosed(EventArgs e)
    {
        BackupService.Instance.Stop();
        ToastService.Dispose();
        base.OnClosed(e);
    }

    private void OnNav(object sender, RoutedEventArgs e)
    {
        if (sender is RadioButton rb && rb.Tag is string page)
            Navigate(page);
    }

    private void Navigate(string page)
    {
        _currentPage = page;

        var (title, sub) = page switch
        {
            "Dashboard" => ("Painel", "Visão geral do processamento"),
            "Documents" => ("Documentos", "Gerenciar documentos fiscais"),
            "Clients" => ("Clientes", "Cadastro de empresas atendidas"),
            "Schemas" => ("Schemas IA", "Modelos de extração por tipo de documento"),
            "Users" => ("Usuários", "Gestão de acessos e permissões"),
            "Settings" => ("Configurações", "Preferências do sistema"),
            _ => ("Painel", "")
        };

        txtPageTitle.Text = title;
        txtPageSub.Text = sub;

        pageHost.Content = page switch
        {
            "Dashboard" => new DashboardPage(),
            "Documents" => new DocumentsPage(_auth),
            "Clients" => new ClientsPage(_auth),
            "Schemas" => new SchemasPage(_auth),
            "Users" => new UsersPage(_auth),
            "Settings" => new SettingsPage(),
            _ => new DashboardPage()
        };
    }

    private void OnToggleTheme(object sender, RoutedEventArgs e)
    {
        ThemeManager.ToggleTheme();
        Navigate(_currentPage);
    }

    private void OnLogout(object sender, RoutedEventArgs e)
    {
        _auth.Logout();
        var login = new LoginView();
        login.Show();
        Close();
    }
}

internal sealed class RelayCommand : System.Windows.Input.ICommand
{
    private readonly Action _exec;
    public RelayCommand(Action exec) { _exec = exec; }
    public event EventHandler? CanExecuteChanged;
    public bool CanExecute(object? parameter) => true;
    public void Execute(object? parameter) => _exec();
}
