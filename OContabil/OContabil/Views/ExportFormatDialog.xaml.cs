using System.Windows;
using System.Windows.Controls;
using OContabil.Services.Exports;

namespace OContabil.Views;

public partial class ExportFormatDialog : Window
{
    public ExportFormat? SelectedFormat { get; private set; }
    public bool OnlyValidated => chkOnlyValidated.IsChecked == true;

    public ExportFormatDialog()
    {
        InitializeComponent();
        lstFormats.SelectedIndex = 0;
    }

    private void OnConfirm(object sender, RoutedEventArgs e)
    {
        if (lstFormats.SelectedItem is ListBoxItem item && item.Tag is string tag &&
            Enum.TryParse<ExportFormat>(tag, out var fmt))
        {
            SelectedFormat = fmt;
            DialogResult = true;
            Close();
        }
        else
        {
            MessageBox.Show("Selecione um formato.", "OContabil", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OnCancel(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}
