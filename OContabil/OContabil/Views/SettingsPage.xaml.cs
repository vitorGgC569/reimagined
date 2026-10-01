using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;
using OContabil.Services;

namespace OContabil.Views;

public partial class SettingsPage : UserControl
{
    private readonly CertificateService _certService = new();
    private readonly GlinerService _glinerService = new();
    private bool _isInitialized;

    public SettingsPage()
    {
        InitializeComponent();
        UpdatePythonStatus();
        LoadFromSettings();
        UpdateTelemetryStats();
        _isInitialized = true;
    }

    private void LoadFromSettings()
    {
        txtCompanyName.Text = AppSettings.CompanyName;
        txtCompanyCnpj.Text = AppSettings.CompanyCnpj;

        SelectByTag(cmbEngine, AppSettings.PreferredEngine.ToString());
        SelectByTag(cmbModel, AppSettings.GlinerModelName);

        txtThreshold.Text = AppSettings.GlinerThreshold.ToString("F2", CultureInfo.InvariantCulture);
        txtTimeout.Text = AppSettings.AiTimeoutSeconds.ToString();
        txtRetries.Text = AppSettings.AiMaxRetries.ToString();

        txtSessionTimeout.Text = AppSettings.SessionTimeoutMinutes.ToString();
        txtStaleHours.Text = AppSettings.StaleDocumentHours.ToString();

        txtBackupFolder.Text = AppSettings.BackupFolder;
        txtBackupInterval.Text = AppSettings.BackupIntervalHours.ToString();
        txtBackupMax.Text = AppSettings.BackupMaxFiles.ToString();

        chkTelemetry.IsChecked = AppSettings.TelemetryOptIn;
    }

    private static void SelectByTag(ComboBox combo, string tag)
    {
        foreach (var i in combo.Items.OfType<ComboBoxItem>())
        {
            if ((i.Tag as string) == tag) { combo.SelectedItem = i; return; }
        }
        if (combo.Items.Count > 0) combo.SelectedIndex = 0;
    }

    private void UpdatePythonStatus()
    {
        if (_glinerService.IsPythonAvailable)
        {
            txtPythonStatus.Text = "Python encontrado ✓";
            txtPythonStatus.Foreground = (System.Windows.Media.Brush)FindResource("Success");
        }
        else
        {
            txtPythonStatus.Text = "Python nao encontrado";
            txtPythonStatus.Foreground = (System.Windows.Media.Brush)FindResource("Error");
        }
    }

    private void UpdateTelemetryStats()
    {
        var snapshot = TelemetryService.Snapshot();
        txtTelemetryStats.Text =
            $"Docs processados: {snapshot.DocumentsProcessed}\n" +
            $"Validados manualmente: {snapshot.DocumentsValidated}\n" +
            $"Falhas IA: {snapshot.AiFailures}\n" +
            $"Exports gerados: {snapshot.ExportsGenerated}\n" +
            $"Última atualização: {snapshot.LastUpdated:dd/MM/yyyy HH:mm}";
    }

    // ── Handlers ───────────────────────────────────────────────────────
    private void OnCompanyChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        AppSettings.CompanyName = txtCompanyName.Text.Trim();
        AppSettings.CompanyCnpj = txtCompanyCnpj.Text.Trim();
    }

    private void OnEngineChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_isInitialized) return;
        if (cmbEngine.SelectedItem is ComboBoxItem item && item.Tag is string tag
            && Enum.TryParse<AiEnginePreference>(tag, out var eng))
        {
            AppSettings.PreferredEngine = eng;
            ToastService.ShowInfo($"Engine preferido: {item.Content}");
        }
    }

    private void OnModelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_isInitialized) return;
        if (cmbModel.SelectedItem is ComboBoxItem item && item.Tag is string modelTag)
        {
            AppSettings.GlinerModelName = modelTag;
        }
    }

    private void OnThresholdChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        if (double.TryParse(txtThreshold.Text.Replace(',', '.'),
            NumberStyles.Float, CultureInfo.InvariantCulture, out double val))
        {
            AppSettings.GlinerThreshold = Math.Clamp(val, 0.0, 1.0);
        }
    }

    private void OnTimeoutChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        if (int.TryParse(txtTimeout.Text, out var t))
            AppSettings.AiTimeoutSeconds = t;
        txtTimeout.Text = AppSettings.AiTimeoutSeconds.ToString();
    }

    private void OnRetriesChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        if (int.TryParse(txtRetries.Text, out var t))
            AppSettings.AiMaxRetries = t;
        txtRetries.Text = AppSettings.AiMaxRetries.ToString();
    }

    private void OnSessionChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        if (int.TryParse(txtSessionTimeout.Text, out var t))
            AppSettings.SessionTimeoutMinutes = t;
        txtSessionTimeout.Text = AppSettings.SessionTimeoutMinutes.ToString();
    }

    private void OnStaleChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        if (int.TryParse(txtStaleHours.Text, out var t))
            AppSettings.StaleDocumentHours = t;
        txtStaleHours.Text = AppSettings.StaleDocumentHours.ToString();
    }

    private void OnBackupChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        AppSettings.BackupFolder = txtBackupFolder.Text.Trim();
        if (int.TryParse(txtBackupInterval.Text, out var i))
            AppSettings.BackupIntervalHours = i;
        if (int.TryParse(txtBackupMax.Text, out var m))
            AppSettings.BackupMaxFiles = m;
        txtBackupInterval.Text = AppSettings.BackupIntervalHours.ToString();
        txtBackupMax.Text = AppSettings.BackupMaxFiles.ToString();
        BackupService.Instance.ConfigureFromSettings();
    }

    private void OnPickBackupFolder(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Selecione a pasta de backup"
        };
        if (dlg.ShowDialog() == true)
        {
            txtBackupFolder.Text = dlg.FolderName;
            OnBackupChanged(sender, e);
        }
    }

    private void OnBackupNow(object sender, RoutedEventArgs e)
    {
        try
        {
            var path = BackupService.Instance.RunNow();
            ToastService.ShowSuccess($"Backup criado:\n{path}");
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Falha no backup: {ex.Message}");
        }
    }

    private void OnTelemetryChanged(object sender, RoutedEventArgs e)
    {
        if (!_isInitialized) return;
        AppSettings.TelemetryOptIn = chkTelemetry.IsChecked == true;
    }

    private void OnResetTelemetry(object sender, RoutedEventArgs e)
    {
        TelemetryService.Reset();
        UpdateTelemetryStats();
        ToastService.ShowSuccess("Contadores zerados.");
    }

    private void OnLoadCert(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "Certificado Digital (*.pfx;*.p12)|*.pfx;*.p12",
            Title = "Selecionar Certificado Digital A1"
        };
        if (dlg.ShowDialog() != true) return;

        var pwdDialog = new PasswordDialog { Owner = Window.GetWindow(this) };
        if (pwdDialog.ShowDialog() != true || string.IsNullOrEmpty(pwdDialog.Password))
            return;

        var result = _certService.LoadCertificate(dlg.FileName, pwdDialog.Password);
        if (result.Success)
        {
            txtCertStatus.Text = "Certificado carregado ✓";
            txtCertStatus.Foreground = (System.Windows.Media.Brush)FindResource("Success");
            txtCertInfo.Text = $"Titular: {result.Subject}\n" +
                              $"Valido ate: {result.ValidUntil:dd/MM/yyyy}\n" +
                              $"Emissor: {result.Issuer}\n" +
                              $"Serial: {result.SerialNumber}";
        }
        else
        {
            txtCertStatus.Text = "Erro ao carregar certificado";
            txtCertStatus.Foreground = (System.Windows.Media.Brush)FindResource("Error");
            txtCertInfo.Text = result.ErrorMessage;
        }
    }
}
