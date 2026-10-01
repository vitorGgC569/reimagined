using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using ContaDocAI.Services;
using Microsoft.Win32;

namespace ContaDocAI.Views;

public partial class UploadView : UserControl
{
    public UploadView()
    {
        InitializeComponent();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        PopulateQueue();
    }

    private void PopulateQueue()
    {
        var items = MockDataService.ProcessingQueue.Select(q =>
        {
            var (icon, label, badgeBg, badgeFg) = q.Status switch
            {
                "processing" => ("⚙️", "Processando",
                    (Brush)Application.Current.FindResource("InfoBgBrush"),
                    (Brush)Application.Current.FindResource("InfoBrush")),
                "completed" => ("✅", "Concluido",
                    (Brush)Application.Current.FindResource("SuccessBgBrush"),
                    (Brush)Application.Current.FindResource("SuccessBrush")),
                _ => ("⏳", "Na Fila",
                    (Brush)Application.Current.FindResource("ElevatedBrush"),
                    (Brush)Application.Current.FindResource("TextTertiaryBrush")),
            };

            return new
            {
                q.Filename,
                q.Size,
                FilesText = $"{q.FilesCount} arquivo{(q.FilesCount > 1 ? "s" : "")}",
                Icon = icon,
                Label = label,
                BadgeBg = badgeBg,
                BadgeFg = badgeFg,
                ShowProgress = q.Status == "processing" ? Visibility.Visible : Visibility.Collapsed,
                ProgressWidth = q.Progress * 1.6, // 160px max
                ProgressStr = $"{q.Progress}%"
            };
        }).ToList();

        queueList.ItemsSource = items;
    }

    private void OnBrowseClick(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Multiselect = true,
            Filter = "Documentos (*.pdf;*.jpg;*.png;*.zip)|*.pdf;*.jpg;*.jpeg;*.png;*.zip|Todos (*.*)|*.*",
            Title = "Selecionar Documentos para Processamento"
        };

        if (dialog.ShowDialog() == true)
        {
            MessageBox.Show($"{dialog.FileNames.Length} arquivo(s) selecionado(s) para processamento!",
                "ContaDoc AI", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }

    private void OnDrop(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            var files = (string[])e.Data.GetData(DataFormats.FileDrop)!;
            MessageBox.Show($"{files.Length} arquivo(s) enviado(s) para processamento!",
                "ContaDoc AI", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        dropZone.BorderBrush = new SolidColorBrush(Color.FromArgb(0x33, 0x63, 0x66, 0xf1));
    }

    private void OnDragOver(object sender, DragEventArgs e)
    {
        e.Effects = DragDropEffects.Copy;
        e.Handled = true;
        dropZone.BorderBrush = (SolidColorBrush)Application.Current.FindResource("AccentBrush");
    }

    private void OnDragLeave(object sender, DragEventArgs e)
    {
        dropZone.BorderBrush = new SolidColorBrush(Color.FromArgb(0x33, 0x63, 0x66, 0xf1));
    }
}
