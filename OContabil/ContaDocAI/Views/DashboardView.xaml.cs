using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using ContaDocAI.Services;

namespace ContaDocAI.Views;

public partial class DashboardView : UserControl
{
    public DashboardView()
    {
        InitializeComponent();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        // Populate chart
        DrawChart();

        // Populate activity
        activityList.ItemsSource = MockDataService.RecentActivity.Select(a => new
        {
            a.Time, a.Action, a.Client, a.Status, a.Doc
        }).ToList();

        // Populate chart labels
        chartLabels.ItemsSource = MockDataService.VolumeChart.Select(v => v.Day).ToList();

        // Populate clients grid
        clientsGrid.ItemsSource = MockDataService.Clients.Take(5).ToList();
    }

    private void DrawChart()
    {
        chartCanvas.Children.Clear();
        var data = MockDataService.VolumeChart;
        int maxCount = data.Max(d => d.Count);
        double canvasWidth = chartCanvas.ActualWidth > 0 ? chartCanvas.ActualWidth : 600;
        double canvasHeight = 180;
        double barWidth = (canvasWidth - (data.Count - 1) * 4) / data.Count;

        for (int i = 0; i < data.Count; i++)
        {
            double barHeight = (double)data[i].Count / maxCount * (canvasHeight - 10);
            var rect = new Rectangle
            {
                Width = barWidth,
                Height = barHeight,
                RadiusX = 3,
                RadiusY = 3,
                Fill = new LinearGradientBrush(
                    Color.FromRgb(99, 102, 241),
                    Color.FromRgb(79, 70, 229),
                    90),
                Opacity = 0.7,
                ToolTip = $"{data[i].Day}: {data[i].Count} docs"
            };

            rect.MouseEnter += (s, e) => ((Rectangle)s!).Opacity = 1;
            rect.MouseLeave += (s, e) => ((Rectangle)s!).Opacity = 0.7;

            Canvas.SetLeft(rect, i * (barWidth + 4));
            Canvas.SetTop(rect, canvasHeight - barHeight);
            chartCanvas.Children.Add(rect);
        }
    }
}
