using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Views;

/// <summary>
/// Assistente exibido no primeiro uso. Coleta dados da empresa, cria o usuário
/// administrador (desabilitando admin/admin) e verifica dependências opcionais
/// (Python, Tesseract). Marca <see cref="AppSettings.FirstRunCompleted"/> ao final.
/// </summary>
public partial class FirstRunWizard : Window
{
    private int _step = 1;
    private readonly GlinerService _gliner = new();

    public FirstRunWizard()
    {
        InitializeComponent();
        txtCompanyName.Text = AppSettings.CompanyName;
        txtCompanyCnpj.Text = AppSettings.CompanyCnpj;
        txtWorkFolder.Text = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
    }

    private void OnPickWorkFolder(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "Selecione a pasta de trabalho" };
        if (dlg.ShowDialog() == true)
            txtWorkFolder.Text = dlg.FolderName;
    }

    private void OnNext(object sender, RoutedEventArgs e)
    {
        switch (_step)
        {
            case 1:
                if (string.IsNullOrWhiteSpace(txtCompanyName.Text))
                {
                    MessageBox.Show("Informe a razão social.", "Validação",
                        MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }
                AppSettings.CompanyName = txtCompanyName.Text.Trim();
                AppSettings.CompanyCnpj = txtCompanyCnpj.Text.Trim();
                GoToStep(2);
                break;

            case 2:
                if (!ValidateAdmin()) return;
                if (!SaveAdminUser()) return;
                GoToStep(3);
                _ = CheckDependenciesAsync();
                break;

            case 3:
                AppSettings.FirstRunCompleted = true;
                AuditLogger.Write(null, "firstrun.completed", "App", null,
                    $"company={AppSettings.CompanyName}");
                DialogResult = true;
                Close();
                break;
        }
    }

    private void OnBack(object sender, RoutedEventArgs e)
    {
        if (_step > 1) GoToStep(_step - 1);
    }

    private void GoToStep(int step)
    {
        _step = step;
        pnlStep1.Visibility = step == 1 ? Visibility.Visible : Visibility.Collapsed;
        pnlStep2.Visibility = step == 2 ? Visibility.Visible : Visibility.Collapsed;
        pnlStep3.Visibility = step == 3 ? Visibility.Visible : Visibility.Collapsed;

        txtStepInfo.Text = step switch
        {
            1 => "Etapa 1 de 3 — Dados da empresa",
            2 => "Etapa 2 de 3 — Usuário administrador",
            3 => "Etapa 3 de 3 — Dependências",
            _ => ""
        };

        btnBack.Visibility = step > 1 ? Visibility.Visible : Visibility.Collapsed;
        btnNext.Content = step == 3 ? "Concluir" : "Avançar";
    }

    private bool ValidateAdmin()
    {
        if (string.IsNullOrWhiteSpace(txtAdminName.Text) ||
            string.IsNullOrWhiteSpace(txtAdminUser.Text))
        {
            MessageBox.Show("Preencha nome completo e usuário.", "Validação",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        if (pwdAdmin.Password.Length < 8)
        {
            MessageBox.Show("A senha precisa ter pelo menos 8 caracteres.", "Validação",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        if (pwdAdmin.Password != pwdAdminConfirm.Password)
        {
            MessageBox.Show("As senhas não conferem.", "Validação",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        if (txtAdminUser.Text.Equals("admin", StringComparison.OrdinalIgnoreCase))
        {
            MessageBox.Show("Escolha um nome de usuário diferente de 'admin'.", "Validação",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        return true;
    }

    private bool SaveAdminUser()
    {
        try
        {
            using var db = new AppDbContext();
            var newUser = new User
            {
                Username = txtAdminUser.Text.Trim(),
                FullName = txtAdminName.Text.Trim(),
                Email = txtAdminEmail.Text.Trim(),
                Role = UserRole.Admin,
                PasswordHash = PasswordHasher.Hash(pwdAdmin.Password),
                IsActive = true,
                CreatedAt = DateTime.Now
            };
            db.Users.Add(newUser);

            // Desativa o admin padrão.
            var defaultAdmin = db.Users.FirstOrDefault(u => u.Username == "admin");
            if (defaultAdmin != null)
            {
                defaultAdmin.IsActive = false;
                defaultAdmin.MustChangePassword = false;
            }

            db.SaveChanges();
            AuditLogger.Write(newUser.Id, "user.firstrun.create", "Users", newUser.Id, newUser.Username);
            return true;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Falha ao criar usuário: {ex.Message}", "Erro",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
    }

    private async System.Threading.Tasks.Task CheckDependenciesAsync()
    {
        // Python
        if (_gliner.IsPythonAvailable)
        {
            txtPythonStatus.Text = "Detectado ✓";
            txtPythonIcon.Text = "✓";
            txtPythonIcon.Foreground = (Brush)FindResource("Success");
        }
        else
        {
            txtPythonStatus.Text = "Não encontrado — modo offline (Regex) ativo";
            txtPythonIcon.Text = "ℹ";
            txtPythonIcon.Foreground = (Brush)FindResource("Warning");
        }

        // Tesseract — best-effort
        bool tessOk = await System.Threading.Tasks.Task.Run(() =>
        {
            try
            {
                var psi = new ProcessStartInfo("tesseract", "--version")
                {
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                var p = Process.Start(psi);
                p!.WaitForExit(3000);
                return p.ExitCode == 0;
            }
            catch { return false; }
        });

        if (tessOk)
        {
            txtTessStatus.Text = "Detectado ✓ — OCR de imagens disponível";
            txtTessIcon.Text = "✓";
            txtTessIcon.Foreground = (Brush)FindResource("Success");
        }
        else
        {
            txtTessStatus.Text = "Não detectado — recomendado para PDFs digitalizados";
            txtTessIcon.Text = "ℹ";
            txtTessIcon.Foreground = (Brush)FindResource("Warning");
        }
    }

    private void OnTesseractHelp(object sender, RoutedEventArgs e)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "https://github.com/UB-Mannheim/tesseract/wiki",
                UseShellExecute = true
            });
        }
        catch { }
    }
}
