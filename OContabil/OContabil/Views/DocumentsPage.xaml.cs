using System.IO;
using System.Windows;
using System.Windows.Controls;
using Microsoft.EntityFrameworkCore;
using Microsoft.Win32;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;
using OContabil.Services.Exports;

namespace OContabil.Views;

public partial class DocumentsPage : UserControl
{
    private readonly AuthService _auth;
    private readonly GlinerService _gliner;
    private List<Document> _allDocs = new();

    public DocumentsPage(AuthService auth)
    {
        InitializeComponent();
        _auth = auth;
        _gliner = new GlinerService();
        
        // Subscribe to queue completion events to auto-refresh the grid
        DocumentProcessingQueue.Instance.DocumentProcessed += OnQueueDocumentProcessed;
    }

    private void OnQueueDocumentProcessed(int docId)
    {
        // This is called on UI thread by the queue
        ReloadDocs();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        cmbType.Items.Clear();
        cmbType.Items.Add("Todos os tipos");
        foreach (var t in new[] { "NF-e", "NFS-e", "CT-e", "Boleto", "DARF", "Comprovante PIX", "Outro" })
            cmbType.Items.Add(t);
        cmbType.SelectedIndex = 0;

        cmbStatus.Items.Clear();
        cmbStatus.Items.Add("Todos os status");
        cmbStatus.Items.Add("Pendente");
        cmbStatus.Items.Add("Processando");
        cmbStatus.Items.Add("Revisar");
        cmbStatus.Items.Add("Validado");
        cmbStatus.Items.Add("Erro");
        cmbStatus.SelectedIndex = 0;

        ReloadDocs();
    }

    private void ReloadDocs()
    {
        try
        {
            using var db = new AppDbContext();
            _allDocs = db.Documents.Include(d => d.Client).OrderByDescending(d => d.UploadedAt).ToList();
            RefreshGrid();
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao carregar documentos: {ex.Message}");
        }
    }

    private void OnFilter(object sender, object e)
    {
        RefreshGrid();
    }

    private void RefreshGrid()
    {
        var query = _allDocs.AsEnumerable();

        var search = txtSearch?.Text?.Trim() ?? "";
        if (!string.IsNullOrEmpty(search))
            query = query.Where(d =>
                d.Filename.Contains(search, StringComparison.OrdinalIgnoreCase) ||
                (d.Client?.Name ?? "").Contains(search, StringComparison.OrdinalIgnoreCase));

        if (cmbType?.SelectedIndex > 0)
        {
            var type = cmbType.SelectedItem?.ToString() ?? "";
            query = query.Where(d => d.DocumentType == type);
        }

        if (cmbStatus?.SelectedIndex > 0)
        {
            var statusText = cmbStatus.SelectedItem?.ToString() ?? "";
            query = query.Where(d => d.StatusDisplay == statusText);
        }

        var list = query.ToList();

        gridDocs.ItemsSource = list;
        if (txtCount != null)
        {
            int queueCount = DocumentProcessingQueue.Instance.PendingCount;
            txtCount.Text = queueCount > 0
                ? $"{list.Count} documento(s) | {queueCount} na fila de processamento"
                : $"{list.Count} documento(s)";
        }
    }

    private void OnImport(object sender, RoutedEventArgs e)
    {
        if (!_auth.CanEdit)
        {
            ToastService.ShowWarning("Sem permissao para importar documentos.");
            return;
        }

        List<Client> clients;
        using (var db = new AppDbContext())
        {
            clients = db.Clients.Where(c => c.IsActive).OrderBy(c => c.Name).ToList();
        }
        
        if (!clients.Any())
        {
            ToastService.ShowInfo("Cadastre pelo menos um cliente antes de importar documentos.");
            return;
        }

        var dlg = new OpenFileDialog
        {
            Multiselect = true,
            Filter = "Documentos (*.pdf;*.xml;*.jpg;*.png;*.txt)|*.pdf;*.xml;*.jpg;*.jpeg;*.png;*.txt|Todos (*.*)|*.*",
            Title = "Importar Documentos"
        };

        if (dlg.ShowDialog() != true) return;

        // Client selection
        Client targetClient;
        if (clients.Count == 1)
        {
            targetClient = clients.First();
        }
        else
        {
            var selectionWindow = new Window
            {
                Title = "Selecionar Cliente",
                Width = 400,
                Height = 300,
                WindowStartupLocation = WindowStartupLocation.CenterScreen,
                Background = (System.Windows.Media.Brush)FindResource("BgPrimary"),
                FontFamily = new System.Windows.Media.FontFamily("Segoe UI"),
                ResizeMode = ResizeMode.NoResize
            };

            var selectedClient = (Client?)null;
            var sp = new StackPanel { Margin = new Thickness(20) };
            sp.Children.Add(new TextBlock 
            { 
                Text = "Selecione o cliente para os documentos:", 
                FontSize = 14, 
                FontWeight = FontWeights.SemiBold,
                Foreground = (System.Windows.Media.Brush)FindResource("TextPrimary"),
                Margin = new Thickness(0, 0, 0, 12) 
            });

            var listBox = new ListBox
            {
                Height = 160,
                FontSize = 12,
                Background = (System.Windows.Media.Brush)FindResource("BgCard"),
                Foreground = (System.Windows.Media.Brush)FindResource("TextPrimary"),
                BorderBrush = (System.Windows.Media.Brush)FindResource("Border")
            };
            foreach (var c in clients)
                listBox.Items.Add(new ListBoxItem { Content = $"{c.Name} ({c.Cnpj})", Tag = c });
            listBox.SelectedIndex = 0;
            sp.Children.Add(listBox);

            var btnOk = new Button 
            { 
                Content = "Confirmar", 
                Style = (Style)FindResource("BtnPrimary"),
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 12, 0, 0),
                Width = 120
            };
            btnOk.Click += (_, _) =>
            {
                if (listBox.SelectedItem is ListBoxItem li && li.Tag is Client cl)
                    selectedClient = cl;
                selectionWindow.DialogResult = true;
                selectionWindow.Close();
            };
            sp.Children.Add(btnOk);

            selectionWindow.Content = sp;
            if (selectionWindow.ShowDialog() != true || selectedClient == null)
                return;

            targetClient = selectedClient;
        }

        // Save documents to DB and enqueue for processing
        int enqueued = 0;
        foreach (var filePath in dlg.FileNames)
        {
            var fileInfo = new FileInfo(filePath);

            try
            {
                using var innerDb = new AppDbContext();
                
                var doc = new Document
                {
                    Filename = fileInfo.Name,
                    FilePath = fileInfo.FullName,
                    DocumentType = GuessDocType(fileInfo.Name),
                    Status = DocumentStatus.Pending,
                    FileSizeBytes = fileInfo.Length,
                    UploadedAt = DateTime.Now,
                    ClientId = targetClient.Id,
                    UploadedByUserId = _auth.CurrentUser!.Id
                };

                innerDb.Documents.Add(doc);
                
                var clientToUpdate = innerDb.Clients.Find(targetClient.Id);
                if (clientToUpdate != null) clientToUpdate.DocumentCount++;
                
                innerDb.SaveChanges();

                // ── ENQUEUE FOR BACKGROUND PROCESSING ──
                DocumentProcessingQueue.Instance.Enqueue(doc.Id, filePath, doc.DocumentType);
                enqueued++;
            }
            catch (Exception ex)
            {
                ToastService.ShowError($"Erro ao salvar documento: {ex.Message}");
            }
        }

        ReloadDocs();
        if (enqueued > 0)
            ToastService.ShowSuccess($"{enqueued} documento(s) importado(s) e enfileirado(s) para processamento.");
    }

    private void OnRowDoubleClick(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        if (sender is System.Windows.Controls.DataGridRow row && row.Item is Document doc)
        {
            try
            {
                var dialog = new DocumentReviewDialog(doc.Id, _auth);
                if (dialog.ShowDialog() == true)
                {
                    ReloadDocs();
                }
            }
            catch (Exception ex)
            {
                ToastService.ShowError($"Erro ao abrir revisão: {ex.InnerException?.Message ?? ex.Message}");
            }
        }
    }

    private void OnReanalyzeDoc(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement el && el.DataContext is Document docModel)
        {
            if (string.IsNullOrEmpty(docModel.FilePath) || !File.Exists(docModel.FilePath))
            {
                ToastService.ShowError($"Arquivo original não encontrado: {docModel.FilePath ?? "caminho desconhecido"}");
                return;
            }

            if (MessageBox.Show($"Deseja re-analisar o documento {docModel.Filename}?", 
                "Reanálise", MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes)
            {
                // ── ENQUEUE FOR BACKGROUND RE-PROCESSING ──
                DocumentProcessingQueue.Instance.Enqueue(docModel.Id, docModel.FilePath, docModel.DocumentType);
                ReloadDocs();
            }
        }
    }

    private void OnDeleteDoc(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement el && el.DataContext is Document docModel)
        {
            try
            {
                if (MessageBox.Show($"Excluir PERMANENTEMENTE o documento {docModel.Filename}?", 
                    "Exclusão", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
                {
                    using var db = new AppDbContext();
                    var doc = db.Documents.Find(docModel.Id);
                    if (doc == null) return;
                    
                    var targetClient = db.Clients.FirstOrDefault(c => c.Id == doc.ClientId);
                    if (targetClient != null)
                    {
                        targetClient.DocumentCount = Math.Max(0, targetClient.DocumentCount - 1);
                    }
                    db.Documents.Remove(doc);
                    db.SaveChanges();
                    
                    ReloadDocs();
                    ToastService.ShowSuccess("Documento excluído.");
                }
            }
            catch (Exception ex)
            {
                ToastService.ShowError($"Erro ao excluir: {ex.Message}");
            }
        }
    }

    private static string GuessDocType(string filename)
    {
        var lower = filename.ToLowerInvariant();
        if (lower.Contains("nfe") || lower.Contains("nota")) return "NF-e";
        if (lower.Contains("nfse") || lower.Contains("servico")) return "NFS-e";
        if (lower.Contains("cte") || lower.Contains("transporte")) return "CT-e";
        if (lower.Contains("boleto")) return "Boleto";
        if (lower.Contains("darf") || lower.Contains("guia")) return "DARF";
        if (lower.Contains("pix") || lower.Contains("comprovante")) return "Comprovante PIX";
        return "Outro";
    }

    private void OnExport(object sender, RoutedEventArgs e)
    {
        if (_allDocs.Count == 0)
        {
            MessageBox.Show("Nao ha documentos para exportar.", "OContabil", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var picker = new ExportFormatDialog { Owner = Window.GetWindow(this) };
        if (picker.ShowDialog() != true || picker.SelectedFormat == null)
            return;

        var format = picker.SelectedFormat.Value;
        var manager = new ExportManager();

        var save = new SaveFileDialog
        {
            Title = $"Exportar ({format})",
            Filter = manager.DefaultFilter(format),
            FileName = $"OContabil_{format}_{DateTime.Now:yyyyMMdd_HHmmss}.{manager.DefaultExtension(format)}"
        };
        if (save.ShowDialog() != true) return;

        try
        {
            var request = new ExportRequest
            {
                OutputPath = save.FileName,
                OnlyValidated = picker.OnlyValidated
            };
            var path = manager.Export(format, request);
            if (string.IsNullOrEmpty(path))
            {
                MessageBox.Show("Nenhum documento encontrado com os filtros aplicados.", "OContabil", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            var result = MessageBox.Show($"Exportacao concluida!\n\nSalvo em:\n{path}\n\nAbrir a pasta?", "OContabil Export", MessageBoxButton.YesNo, MessageBoxImage.Information);
            if (result == MessageBoxResult.Yes)
            {
                try
                {
                    System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "explorer.exe",
                        Arguments = $"/select,\"{path}\"",
                        UseShellExecute = true
                    });
                }
                catch { }
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Erro ao exportar: {ex.Message}", "Erro", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}
