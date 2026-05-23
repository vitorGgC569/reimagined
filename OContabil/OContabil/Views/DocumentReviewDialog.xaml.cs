using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;

namespace OContabil.Views;

/// <summary>
/// Diálogo de revisão de documento em três abas: visualização do original,
/// edição de campos extraídos pela IA e histórico de revisões. Toda alteração
/// gera um registro em <see cref="DocumentRevision"/> com diff JSON.
/// </summary>
public partial class DocumentReviewDialog : Window
{
    private readonly int _docId;
    private readonly AuthService? _auth;
    private Document _doc = null!;
    private readonly Dictionary<string, TextBox> _fieldEditors = new();
    private bool _suppressStatusChange;

    public DocumentReviewDialog(int docId, AuthService? auth = null)
    {
        InitializeComponent();
        _docId = docId;
        _auth = auth;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            using var db = new AppDbContext();
            _doc = db.Documents
                .Include(d => d.UploadedBy)
                .Include(d => d.Client)
                .FirstOrDefault(d => d.Id == _docId)!;

            if (_doc == null)
            {
                MessageBox.Show("Documento não encontrado.", "OContabil", MessageBoxButton.OK, MessageBoxImage.Error);
                Close();
                return;
            }

            txtFilename.Text = _doc.Filename;
            txtType.Text = _doc.DocumentType;
            txtStatus.Text = _doc.StatusDisplay;
            txtConfidence.Text = _doc.ConfidenceScore.HasValue
                ? $"Confiança: {_doc.ConfidenceScore.Value:P0}"
                : "Confiança: N/A";

            txtUploadedBy.Text = _doc.UploadedBy != null
                ? $"Enviado por {_doc.UploadedBy.FullName} em {_doc.UploadedAt:dd/MM/yyyy HH:mm}"
                : $"Enviado em {_doc.UploadedAt:dd/MM/yyyy HH:mm}";

            txtMetadata.Text = BuildMetadataText(_doc);

            txtOcr.Text = _doc.OcrText ?? "Nenhum texto extraído.";
            txtJson.Text = PrettyJson(_doc.ExtractedJson) ?? "{}";

            ApplyStatusStyling();
            _suppressStatusChange = true;
            foreach (ComboBoxItem item in cmbStatus.Items)
            {
                if (item.Tag?.ToString() == ((int)_doc.Status).ToString())
                {
                    cmbStatus.SelectedItem = item;
                    break;
                }
            }
            _suppressStatusChange = false;

            UpdateReasonVisibility();
            if (!string.IsNullOrEmpty(_doc.RejectionReason))
                txtReason.Text = _doc.RejectionReason;

            BuildFieldsForm(_doc.ExtractedJson);
            LoadHistory(db);

            _ = InitializeWebViewAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Erro ao carregar documento: {ex.Message}", "OContabil", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void ApplyStatusStyling()
    {
        badgeStatus.Background = _doc.Status switch
        {
            DocumentStatus.Validated => (Brush)FindResource("SuccessBg"),
            DocumentStatus.Error or DocumentStatus.Rejected => (Brush)FindResource("ErrorBg"),
            DocumentStatus.ReadyForReview => (Brush)FindResource("WarningBg"),
            _ => (Brush)FindResource("InfoBg")
        };
        txtStatus.Foreground = _doc.Status switch
        {
            DocumentStatus.Validated => (Brush)FindResource("Success"),
            DocumentStatus.Error or DocumentStatus.Rejected => (Brush)FindResource("Error"),
            DocumentStatus.ReadyForReview => (Brush)FindResource("Warning"),
            _ => (Brush)FindResource("Info")
        };
    }

    private static string BuildMetadataText(Document d)
    {
        var parts = new List<string>();
        parts.Add($"Tamanho: {d.FileSizeDisplay}");
        if (d.ProcessedAt.HasValue) parts.Add($"Processado: {d.ProcessedAt:dd/MM HH:mm}");
        if (d.LastEditedAt.HasValue) parts.Add($"Editado: {d.LastEditedAt:dd/MM HH:mm}");
        if (d.Client != null) parts.Add($"Cliente: {d.Client.Name}");
        return string.Join("  •  ", parts);
    }

    private void OnStatusChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressStatusChange) return;
        UpdateReasonVisibility();
    }

    private void UpdateReasonVisibility()
    {
        var status = ReadSelectedStatus();
        pnlReason.Visibility = status is DocumentStatus.Rejected or DocumentStatus.Error
            ? Visibility.Visible
            : Visibility.Collapsed;
    }

    private DocumentStatus ReadSelectedStatus()
    {
        if (cmbStatus.SelectedItem is ComboBoxItem item &&
            int.TryParse(item.Tag?.ToString(), out int v))
            return (DocumentStatus)v;
        return _doc?.Status ?? DocumentStatus.Pending;
    }

    private async System.Threading.Tasks.Task InitializeWebViewAsync()
    {
        try
        {
            await docViewer.EnsureCoreWebView2Async(null);
            if (!string.IsNullOrEmpty(_doc.FilePath) && System.IO.File.Exists(_doc.FilePath))
            {
                docViewer.Source = new Uri(_doc.FilePath);
            }
            else
            {
                docViewer.NavigateToString(
                    "<html><body style='font-family:Segoe UI; padding: 20px; color:#666;'>" +
                    "<h2>Arquivo não encontrado</h2>" +
                    $"<p>O arquivo original não está mais no caminho:<br/>{(_doc.FilePath ?? "(desconhecido)")}</p>" +
                    "</body></html>");
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                $"O visualizador requer Microsoft Edge WebView2.\n\nDetalhe: {ex.Message}",
                "Componente Ausente", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    // ── Campos editáveis ───────────────────────────────────────────────
    private void BuildFieldsForm(string? json)
    {
        pnlFields.Children.Clear();
        _fieldEditors.Clear();

        if (string.IsNullOrWhiteSpace(json))
        {
            pnlFields.Children.Add(new TextBlock
            {
                Text = "Sem campos extraídos. Reprocesse o documento ou edite o JSON manualmente na lateral.",
                Foreground = (Brush)FindResource("TextMuted"),
                FontSize = 11,
                TextWrapping = TextWrapping.Wrap
            });
            return;
        }

        Dictionary<string, JsonElement>? root = null;
        try
        {
            var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind == JsonValueKind.Object)
            {
                root = doc.RootElement.EnumerateObject()
                    .ToDictionary(p => p.Name, p => p.Value.Clone());
            }
        }
        catch
        {
            pnlFields.Children.Add(new TextBlock
            {
                Text = "JSON inválido. Corrija manualmente na lateral antes de editar campos.",
                Foreground = (Brush)FindResource("Error"),
                FontSize = 11,
                TextWrapping = TextWrapping.Wrap
            });
            return;
        }

        if (root == null) return;

        foreach (var (key, value) in root)
        {
            var label = new TextBlock
            {
                Text = HumanizeFieldName(key),
                FontWeight = FontWeights.SemiBold,
                Foreground = (Brush)FindResource("TextSecondary"),
                FontSize = 11.5,
                Margin = new Thickness(0, 8, 0, 4)
            };
            pnlFields.Children.Add(label);

            var isComplex = value.ValueKind is JsonValueKind.Object or JsonValueKind.Array;
            var textBox = new TextBox
            {
                Style = (Style)FindResource("InputBox"),
                FontSize = 12,
                IsReadOnly = isComplex,
                Text = isComplex
                    ? value.GetRawText()
                    : value.ToString(),
                AcceptsReturn = isComplex,
                TextWrapping = isComplex ? TextWrapping.Wrap : TextWrapping.NoWrap,
                Tag = value.ValueKind
            };
            if (isComplex) textBox.Height = 60;
            pnlFields.Children.Add(textBox);
            _fieldEditors[key] = textBox;
        }
    }

    private static string HumanizeFieldName(string raw)
    {
        var parts = raw.Replace('_', ' ').Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return string.Join(' ', parts.Select(p => char.ToUpper(p[0]) + p.Substring(1).ToLower()));
    }

    private string SerializeFieldsToJson()
    {
        if (_fieldEditors.Count == 0)
            return txtJson.Text;

        using var ms = new System.IO.MemoryStream();
        using var writer = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        foreach (var (key, tb) in _fieldEditors)
        {
            writer.WritePropertyName(key);
            var kind = tb.Tag as JsonValueKind? ?? JsonValueKind.String;
            try
            {
                if (kind is JsonValueKind.Object or JsonValueKind.Array)
                {
                    using var nested = JsonDocument.Parse(tb.Text);
                    nested.RootElement.WriteTo(writer);
                }
                else if (kind == JsonValueKind.Number && decimal.TryParse(
                    tb.Text.Replace(',', '.'),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var n))
                {
                    writer.WriteNumberValue(n);
                }
                else if (kind == JsonValueKind.True || kind == JsonValueKind.False)
                {
                    writer.WriteBooleanValue(bool.TryParse(tb.Text, out var b) && b);
                }
                else if (kind == JsonValueKind.Null && string.IsNullOrEmpty(tb.Text))
                {
                    writer.WriteNullValue();
                }
                else
                {
                    writer.WriteStringValue(tb.Text);
                }
            }
            catch
            {
                writer.WriteStringValue(tb.Text);
            }
        }
        writer.WriteEndObject();
        writer.Flush();
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }

    private static string? PrettyJson(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return raw;
        try
        {
            var doc = JsonDocument.Parse(raw);
            return JsonSerializer.Serialize(doc.RootElement, new JsonSerializerOptions { WriteIndented = true });
        }
        catch
        {
            return raw;
        }
    }

    // ── Histórico ──────────────────────────────────────────────────────
    private void LoadHistory(AppDbContext db)
    {
        var revisions = db.DocumentRevisions
            .Where(r => r.DocumentId == _docId)
            .OrderByDescending(r => r.RevisedAt)
            .ToList();

        var users = db.Users.ToDictionary(u => u.Id, u => u.FullName);

        gridHistory.ItemsSource = revisions.Select(r => new
        {
            RevisedAtStr = r.RevisedAt.ToString("dd/MM/yyyy HH:mm"),
            UserName = users.TryGetValue(r.UserId ?? 0, out var n) ? n : $"#{r.UserId}",
            FromStatus = StatusLabel(r.PreviousStatus),
            ToStatus = StatusLabel(r.NewStatus),
            Reason = r.Reason ?? ""
        }).ToList();
    }

    private static string StatusLabel(DocumentStatus s) => s switch
    {
        DocumentStatus.Pending => "Pendente",
        DocumentStatus.Processing => "Processando",
        DocumentStatus.ReadyForReview => "Revisar",
        DocumentStatus.Validated => "Validado",
        DocumentStatus.Rejected => "Rejeitado",
        DocumentStatus.Error => "Erro",
        _ => s.ToString()
    };

    // ── Ações ──────────────────────────────────────────────────────────
    private void OnRestoreOriginal(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_doc.ExtractedJsonOriginal))
        {
            ToastService.ShowInfo("Não há JSON original preservado para restaurar.");
            return;
        }
        if (MessageBox.Show("Descartar edições e restaurar o JSON original da IA?",
                "Restaurar", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes)
            return;

        txtJson.Text = PrettyJson(_doc.ExtractedJsonOriginal);
        BuildFieldsForm(_doc.ExtractedJsonOriginal);
    }

    private void OnSave(object sender, RoutedEventArgs e)
    {
        try
        {
            var newStatus = ReadSelectedStatus();
            string? reason = null;
            if (newStatus is DocumentStatus.Rejected or DocumentStatus.Error)
            {
                reason = txtReason.Text?.Trim();
                if (string.IsNullOrEmpty(reason))
                {
                    MessageBox.Show("Informe o motivo da rejeição/erro antes de salvar.",
                        "Validação", MessageBoxButton.OK, MessageBoxImage.Warning);
                    txtReason.Focus();
                    return;
                }
            }

            // Reconstroi JSON com base nos campos editáveis (se houver), senão usa o textarea
            string newJson = _fieldEditors.Count > 0 ? SerializeFieldsToJson() : txtJson.Text;
            try { JsonDocument.Parse(newJson); }
            catch (Exception ex)
            {
                MessageBox.Show($"JSON inválido: {ex.Message}", "Validação",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            using var db = new AppDbContext();
            var docToUpdate = db.Documents.FirstOrDefault(d => d.Id == _docId);
            if (docToUpdate == null) return;

            var prevStatus = docToUpdate.Status;
            var prevJson = docToUpdate.ExtractedJson;

            // Preserva original na primeira edição
            if (string.IsNullOrEmpty(docToUpdate.ExtractedJsonOriginal))
                docToUpdate.ExtractedJsonOriginal = prevJson;

            docToUpdate.ExtractedJson = newJson;
            docToUpdate.LastEditedAt = DateTime.Now;
            docToUpdate.Status = newStatus;
            docToUpdate.RejectionReason = reason;

            if (newStatus == DocumentStatus.Validated)
            {
                docToUpdate.ConfidenceScore = 1.0;
                docToUpdate.ValidatedAt = DateTime.Now;
                docToUpdate.ValidatedByUserId = _auth?.CurrentUser?.Id;
            }

            // Registra revisão
            var userId = _auth?.CurrentUser?.Id ?? 0;
            db.DocumentRevisions.Add(new DocumentRevision
            {
                DocumentId = _docId,
                UserId = userId,
                RevisedAt = DateTime.Now,
                PreviousJson = prevJson,
                NewJson = newJson,
                PreviousStatus = prevStatus,
                NewStatus = newStatus,
                Reason = reason
            });
            db.SaveChanges();

            AuditLogger.Write(
                userId == 0 ? null : userId,
                "document.review",
                "Document",
                _docId,
                $"{prevStatus} → {newStatus}{(reason != null ? $" ({reason})" : "")}");

            ToastService.ShowSuccess("Alterações salvas.");
            DialogResult = true;
            Close();
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao salvar: {ex.Message}");
        }
    }

    private void OnCancel(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}
