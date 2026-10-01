using System.Windows;

namespace OContabil.Views;

public partial class PasswordDialog : Window
{
    public string Password => txtPassword.Password;

    public PasswordDialog()
    {
        InitializeComponent();
        txtPassword.Focus();
    }

    private void OnOk(object sender, RoutedEventArgs e)
    {
        DialogResult = true;
        Close();
    }

    private void OnCancel(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}
