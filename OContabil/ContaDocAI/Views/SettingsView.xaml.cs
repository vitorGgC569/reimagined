using System.Windows;
using System.Windows.Controls;

namespace ContaDocAI.Views;

public partial class SettingsView : UserControl
{
    public SettingsView()
    {
        InitializeComponent();
    }

    private void OnThresholdChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (thresholdText != null)
            thresholdText.Text = $"{(int)e.NewValue}%";
    }
}
