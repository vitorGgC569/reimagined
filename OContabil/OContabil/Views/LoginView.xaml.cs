using System.Windows;
using System.Windows.Input;
using OContabil.Data;
using OContabil.Services;

namespace OContabil.Views;

public partial class LoginView : Window
{
    private readonly AuthService _auth;

    public LoginView()
    {
        InitializeComponent();

        using (var db = new AppDbContext())
        {
            DbInitializer.Initialize(db);
        }

        _auth = new AuthService();

        txtPass.KeyDown += (_, e) => { if (e.Key == Key.Enter) OnLogin(this, new RoutedEventArgs()); };
        txtUser.KeyDown += (_, e) => { if (e.Key == Key.Enter) txtPass.Focus(); };

        Loaded += LoginView_Loaded;
    }

    private void LoginView_Loaded(object sender, RoutedEventArgs e)
    {
        Loaded -= LoginView_Loaded;
        if (!AppSettings.FirstRunCompleted)
        {
            var wizard = new FirstRunWizard { Owner = this };
            wizard.ShowDialog();
        }
    }

    private void OnLogin(object sender, RoutedEventArgs e)
    {
        var user = txtUser.Text.Trim();
        var pass = txtPass.Password;

        if (string.IsNullOrEmpty(user) || string.IsNullOrEmpty(pass))
        {
            ShowError("Preencha usuário e senha.");
            return;
        }

        var outcome = _auth.Login(user, pass);
        switch (outcome.Status)
        {
            case LoginStatus.Success:
                var main = new MainWindow(_auth);
                main.Show();
                Close();
                break;

            case LoginStatus.Throttled:
                var minutes = Math.Max(1, (int)Math.Ceiling(outcome.BlockedFor.TotalMinutes));
                ShowError($"Excesso de tentativas. Tente novamente em {minutes} min.");
                txtPass.Password = "";
                break;

            case LoginStatus.InvalidCredentials:
            default:
                var msg = outcome.RemainingAttempts > 0 && outcome.RemainingAttempts < 5
                    ? $"Usuário ou senha incorretos. ({outcome.RemainingAttempts} tentativa(s) restante(s))"
                    : "Usuário ou senha incorretos.";
                ShowError(msg);
                txtPass.Password = "";
                txtPass.Focus();
                break;
        }
    }

    private void ShowError(string msg)
    {
        txtError.Text = msg;
        txtError.Visibility = Visibility.Visible;
    }
}
