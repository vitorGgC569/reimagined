using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Views;

public partial class SchemasPage : UserControl
{
    private readonly AuthService _auth;
    private List<SchemaVm> _schemas = new();
    private SchemaVm? _current;
    private bool _loading;

    public SchemasPage(AuthService auth)
    {
        InitializeComponent();
        _auth = auth;
    }

    private void OnLoaded(object sender, RoutedEventArgs e) => Reload();

    private void Reload()
    {
        try
        {
            _loading = true;
            using var db = new AppDbContext();
            var schemas = db.DocumentSchemas.OrderBy(s => s.IsSystem ? 0 : 1).ThenBy(s => s.Name).ToList();
            _schemas = schemas.Select(s => new SchemaVm
            {
                Id = s.Id,
                Name = s.Name,
                DocumentType = s.DocumentType,
                ClientId = s.ClientId,
                SchemaJson = s.SchemaJson,
                IsSystem = s.IsSystem
            }).ToList();
            lstSchemas.ItemsSource = _schemas;

            var clients = db.Clients.Where(c => c.IsActive).OrderBy(c => c.Name).ToList();
            var withNull = new List<Client> { new Client { Id = 0, Name = "— Todos os clientes —" } };
            withNull.AddRange(clients);
            cmbClient.ItemsSource = withNull;
            cmbClient.SelectedValue = 0;
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao carregar schemas: {ex.Message}");
        }
        finally { _loading = false; }
    }

    private void OnSelected(object sender, SelectionChangedEventArgs e)
    {
        if (_loading) return;
        if (lstSchemas.SelectedItem is SchemaVm vm)
        {
            _current = vm;
            txtName.Text = vm.Name;
            foreach (ComboBoxItem item in cmbDocType.Items)
            {
                if ((item.Content as string) == vm.DocumentType)
                {
                    cmbDocType.SelectedItem = item;
                    break;
                }
            }
            cmbClient.SelectedValue = vm.ClientId ?? 0;
            txtJson.Text = PrettyJson(vm.SchemaJson) ?? vm.SchemaJson;
            txtName.IsEnabled = !vm.IsSystem;
        }
    }

    private void OnNew(object sender, RoutedEventArgs e)
    {
        _current = null;
        txtName.Text = "";
        txtName.IsEnabled = true;
        cmbDocType.SelectedIndex = 0;
        cmbClient.SelectedValue = 0;
        txtJson.Text = "{\n  \"fields\": []\n}";
        lstSchemas.SelectedItem = null;
    }

    private void OnSave(object sender, RoutedEventArgs e)
    {
        if (!_auth.CanManageUsers && _auth.CurrentUser?.Role != UserRole.Admin)
        {
            ToastService.ShowWarning("Apenas administradores podem editar schemas.");
            return;
        }

        if (string.IsNullOrWhiteSpace(txtName.Text))
        {
            ToastService.ShowWarning("Informe um nome para o schema.");
            return;
        }
        if (cmbDocType.SelectedItem is not ComboBoxItem docTypeItem)
        {
            ToastService.ShowWarning("Selecione o tipo de documento.");
            return;
        }

        try { JsonDocument.Parse(txtJson.Text); }
        catch (Exception ex)
        {
            ToastService.ShowError($"JSON inválido: {ex.Message}");
            return;
        }

        try
        {
            using var db = new AppDbContext();
            DocumentSchema entity;
            if (_current?.Id > 0)
            {
                entity = db.DocumentSchemas.First(s => s.Id == _current.Id);
                if (entity.IsSystem)
                {
                    // Sistema: só permite editar o JSON, não nome/tipo.
                    entity.SchemaJson = txtJson.Text;
                }
                else
                {
                    entity.Name = txtName.Text.Trim();
                    entity.DocumentType = (string)docTypeItem.Content;
                    entity.ClientId = (int?)cmbClient.SelectedValue is int cid && cid > 0 ? cid : null;
                    entity.SchemaJson = txtJson.Text;
                }
            }
            else
            {
                entity = new DocumentSchema
                {
                    Name = txtName.Text.Trim(),
                    DocumentType = (string)docTypeItem.Content,
                    ClientId = (int?)cmbClient.SelectedValue is int cid && cid > 0 ? cid : null,
                    SchemaJson = txtJson.Text,
                    IsSystem = false,
                    CreatedAt = DateTime.Now
                };
                db.DocumentSchemas.Add(entity);
            }
            db.SaveChanges();

            AuditLogger.Write(_auth.CurrentUser?.Id, "schema.save", "DocumentSchema", entity.Id, entity.Name);
            ToastService.ShowSuccess("Schema salvo.");
            Reload();
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao salvar: {ex.Message}");
        }
    }

    private void OnDelete(object sender, RoutedEventArgs e)
    {
        if (_current == null || _current.Id == 0)
        {
            ToastService.ShowInfo("Selecione um schema para excluir.");
            return;
        }
        if (_current.IsSystem)
        {
            ToastService.ShowWarning("Schemas de sistema não podem ser excluídos.");
            return;
        }
        if (MessageBox.Show($"Excluir o schema '{_current.Name}'?", "Confirmar",
                MessageBoxButton.YesNo, MessageBoxImage.Warning) != MessageBoxResult.Yes)
            return;

        try
        {
            using var db = new AppDbContext();
            var entity = db.DocumentSchemas.First(s => s.Id == _current.Id);
            db.DocumentSchemas.Remove(entity);
            db.SaveChanges();
            AuditLogger.Write(_auth.CurrentUser?.Id, "schema.delete", "DocumentSchema", entity.Id, entity.Name);
            ToastService.ShowSuccess("Schema excluído.");
            _current = null;
            Reload();
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao excluir: {ex.Message}");
        }
    }

    private static string? PrettyJson(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return raw;
        try
        {
            var doc = JsonDocument.Parse(raw);
            return JsonSerializer.Serialize(doc.RootElement, new JsonSerializerOptions { WriteIndented = true });
        }
        catch { return raw; }
    }

    private sealed class SchemaVm
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public string DocumentType { get; set; } = "";
        public int? ClientId { get; set; }
        public string SchemaJson { get; set; } = "";
        public bool IsSystem { get; set; }
        public Visibility IsSystemVisibility => IsSystem ? Visibility.Visible : Visibility.Collapsed;
    }
}
