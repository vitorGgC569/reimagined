using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Views;

public partial class NewClientDialog : Window
{
    private readonly BrasilApiService _api = new();
    public Client? CreatedClient { get; private set; }
    private CnpjResult? _lookupResult;

    public NewClientDialog()
    {
        InitializeComponent();
    }

    private void OnCnpjLostFocus(object sender, RoutedEventArgs e)
    {
        var raw = txtCnpj.Text.Trim();
        if (string.IsNullOrEmpty(raw)) return;

        if (Validators.ValidateCnpj(raw))
        {
            txtCnpj.Text = Validators.FormatCnpj(raw);
            txtCnpjHint.Text = "CNPJ valido ✓ — clique Consultar para preencher automaticamente";
            txtCnpjHint.Foreground = (Brush)FindResource("Success");
        }
        else
        {
            txtCnpjHint.Text = "CNPJ invalido — verifique os digitos";
            txtCnpjHint.Foreground = (Brush)FindResource("Error");
        }
    }

    private async void OnLookup(object sender, RoutedEventArgs e)
    {
        var cnpj = txtCnpj.Text.Trim();

        if (!Validators.ValidateCnpj(cnpj))
        {
            txtCnpjHint.Text = "Informe um CNPJ valido antes de consultar";
            txtCnpjHint.Foreground = (Brush)FindResource("Error");
            return;
        }

        // Show loading
        btnLookup.IsEnabled = false;
        btnLookup.Content = "Buscando...";
        txtLoading.Visibility = Visibility.Visible;
        pnlCompanyInfo.Visibility = Visibility.Collapsed;

        try
        {
            _lookupResult = await _api.LookupCnpjAsync(cnpj);
        }
        catch (Exception ex)
        {
            _lookupResult = new CnpjResult { RazaoSocial = $"ERRO_API: {ex.Message}" };
        }

        txtLoading.Visibility = Visibility.Collapsed;
        btnLookup.IsEnabled = true;
        btnLookup.Content = "Consultar";

        if (_lookupResult == null)
        {
            txtCnpjHint.Text = "Nao foi possivel consultar. Verifique a conexao com a internet e tente novamente.";
            txtCnpjHint.Foreground = (Brush)FindResource("Warning");
            return;
        }

        if (_lookupResult.RazaoSocial.StartsWith("ERRO_API:"))
        {
            txtCnpjHint.Text = _lookupResult.RazaoSocial;
            txtCnpjHint.Foreground = (Brush)FindResource("Error");
            return;
        }

        // Show company info card
        txtRazaoSocial.Text = _lookupResult.RazaoSocial;
        txtFantasia.Text = !string.IsNullOrWhiteSpace(_lookupResult.NomeFantasia)
            ? _lookupResult.NomeFantasia : "";

        txtSituacao.Text = _lookupResult.SituacaoCadastral;
        txtSituacao.Foreground = _lookupResult.IsAtiva
            ? (Brush)FindResource("Success")
            : (Brush)FindResource("Error");

        txtAtividade.Text = $"Atividade: {_lookupResult.AtividadePrincipal}";
        txtEndereco.Text = $"Endereco: {_lookupResult.EnderecoCompleto}";
        txtCapital.Text = $"Capital: R$ {_lookupResult.CapitalSocial:N2} | {_lookupResult.Porte} | {_lookupResult.NaturezaJuridica}";

        pnlCompanyInfo.Visibility = Visibility.Visible;

        // Auto-fill form fields
        txtName.Text = _lookupResult.RazaoSocial;
        txtEmail.Text = _lookupResult.Email ?? "";
        txtPhone.Text = _lookupResult.TelefoneFormatado;

        // Auto-select regime
        var regime = _lookupResult.RegimeEstimado;
        for (int i = 0; i < cmbRegime.Items.Count; i++)
        {
            if (cmbRegime.Items[i] is ComboBoxItem item &&
                item.Content?.ToString() == regime)
            {
                cmbRegime.SelectedIndex = i;
                break;
            }
        }

        txtCnpjHint.Text = "Dados preenchidos automaticamente via Receita Federal ✓";
        txtCnpjHint.Foreground = (Brush)FindResource("Success");
    }

    private void OnSave(object sender, RoutedEventArgs e)
    {
        HideError();

        var name = txtName.Text.Trim();
        var cnpj = txtCnpj.Text.Trim();
        var email = txtEmail.Text.Trim();
        var phone = txtPhone.Text.Trim();

        if (string.IsNullOrEmpty(name))
        { ShowError("Informe a razao social."); return; }

        if (string.IsNullOrEmpty(cnpj))
        { ShowError("Informe o CNPJ."); return; }

        if (!Validators.ValidateCnpj(cnpj))
        { ShowError("CNPJ invalido."); return; }

        if (!string.IsNullOrEmpty(email) && !Validators.ValidateEmail(email))
        { ShowError("Email invalido."); return; }

        if (!string.IsNullOrEmpty(phone) && !Validators.ValidatePhone(phone))
        { ShowError("Telefone invalido."); return; }

        try
        {
            using var db = new AppDbContext();
            
            var formattedCnpj = Validators.FormatCnpj(cnpj);
            if (db.Clients.Any(c => c.Cnpj == formattedCnpj))
            { ShowError("CNPJ ja cadastrado."); return; }

            if (_lookupResult != null && !_lookupResult.IsAtiva)
            {
                var answer = MessageBox.Show(
                    $"Atencao: esta empresa esta com situacao '{_lookupResult.SituacaoCadastral}'.\n\nDeseja cadastrar mesmo assim?",
                    "Situacao Cadastral", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                if (answer != MessageBoxResult.Yes) return;
            }

            var regime = (cmbRegime.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Simples Nacional";

            var client = new Client
            {
                Name = name,
                Cnpj = formattedCnpj,
                Email = email,
                Phone = !string.IsNullOrEmpty(phone) ? Validators.FormatPhone(phone) : "",
                TaxRegime = regime,
                IsActive = true
            };

            db.Clients.Add(client);
            db.SaveChanges();
            CreatedClient = client;
            DialogResult = true;
            Close();
        }
        catch (Exception ex)
        {
            ShowError($"Erro ao salvar: {ex.Message}");
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

    private void HideError() => txtError.Visibility = Visibility.Collapsed;
}
