using System.Windows;
using System.Windows.Controls;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;
using OContabil.Services;
using SkiaSharp;

namespace OContabil.Views;

/// <summary>
/// Painel de visão geral: indicadores, gráficos LiveChartsCore e alertas de
/// documentos parados. Os dados são lidos a cada Load para refletir mudanças
/// recentes na fila de processamento.
/// </summary>
public partial class DashboardPage : UserControl
{
    public DashboardPage()
    {
        InitializeComponent();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            using var db = new AppDbContext();
            var docs = db.Documents.Include(d => d.Client).OrderByDescending(d => d.UploadedAt).ToList();
            var clients = db.Clients.Where(c => c.IsActive).ToList();

            UpdateStatCards(docs, clients);
            UpdateStaleAlert(docs);
            BuildStatusChart(docs);
            BuildClientsChart(docs);
            BuildConfidenceChart(docs);

            gridRecent.ItemsSource = docs.Take(10).Select(d => new
            {
                d.Filename,
                d.DocumentType,
                ClientName = d.Client?.Name ?? "",
                d.StatusDisplay,
                DateStr = d.UploadedAt.ToString("dd/MM/yyyy"),
                SizeStr = d.FileSizeDisplay
            }).ToList();

            gridClients.ItemsSource = clients;
        }
        catch (Exception ex)
        {
            ToastService.ShowError($"Erro ao carregar painel: {ex.Message}");
        }
    }

    private void UpdateStatCards(List<Document> docs, List<Client> clients)
    {
        int total = docs.Count;
        int validated = docs.Count(d => d.Status == DocumentStatus.Validated);
        int pending = docs.Count(d => d.Status is DocumentStatus.ReadyForReview or DocumentStatus.Pending);

        statDocs.Text = total.ToString();
        statValidated.Text = validated.ToString();
        statValPct.Text = total > 0 ? $"{validated * 100 / total}% do total" : "";
        statPending.Text = pending.ToString();

        var withConfidence = docs.Where(d => d.ConfidenceScore.HasValue).ToList();
        if (withConfidence.Any())
        {
            var avg = withConfidence.Average(d => d.ConfidenceScore!.Value);
            statConfidence.Text = $"{avg:P0}";
        }
        else
        {
            statConfidence.Text = "—";
        }
        statClients.Text = $"{clients.Count} clientes ativos";
    }

    private void UpdateStaleAlert(List<Document> docs)
    {
        var threshold = AppSettings.StaleDocumentHours;
        var cutoff = DateTime.Now.AddHours(-threshold);
        var stale = docs.Where(d =>
            (d.Status == DocumentStatus.Pending || d.Status == DocumentStatus.Processing || d.Status == DocumentStatus.ReadyForReview)
            && d.UploadedAt <= cutoff).ToList();

        if (stale.Count == 0)
        {
            pnlStaleAlert.Visibility = Visibility.Collapsed;
            return;
        }

        pnlStaleAlert.Visibility = Visibility.Visible;
        txtStaleTitle.Text = $"{stale.Count} documento(s) parado(s) há mais de {threshold}h";
        var oldest = stale.OrderBy(d => d.UploadedAt).First();
        var oldestAge = (DateTime.Now - oldest.UploadedAt).TotalHours;
        txtStaleSubtitle.Text = $"Mais antigo: {oldest.Filename} ({oldestAge:F0}h atrás). Verifique a fila de processamento.";
    }

    private void BuildStatusChart(List<Document> docs)
    {
        var statusCounts = Enum.GetValues<DocumentStatus>()
            .Select(s => (Status: s, Count: docs.Count(d => d.Status == s)))
            .Where(x => x.Count > 0)
            .ToList();

        chartStatus.Series = statusCounts.Select(s =>
        {
            var color = StatusColor(s.Status);
            return (ISeries)new PieSeries<double>
            {
                Values = new double[] { s.Count },
                Name = DisplayName(s.Status),
                Fill = new SolidColorPaint(color),
                DataLabelsPaint = new SolidColorPaint(SKColors.White),
                DataLabelsSize = 12
            };
        }).ToArray();
    }

    private void BuildClientsChart(List<Document> docs)
    {
        var top = docs.Where(d => d.Client != null)
            .GroupBy(d => d.Client!.Name)
            .Select(g => new { Name = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .Take(5)
            .ToList();

        if (top.Count == 0)
        {
            chartClients.Series = Array.Empty<ISeries>();
            chartClients.XAxes = new[] { new Axis() };
            return;
        }

        chartClients.Series = new ISeries[]
        {
            new ColumnSeries<int>
            {
                Values = top.Select(t => t.Count).ToArray(),
                Name = "Documentos",
                Fill = new SolidColorPaint(SKColor.Parse("#3B82F6")),
                DataLabelsPaint = new SolidColorPaint(SKColors.White),
                DataLabelsSize = 11,
                DataLabelsPosition = LiveChartsCore.Measure.DataLabelsPosition.Top
            }
        };
        chartClients.XAxes = new Axis[]
        {
            new Axis
            {
                Labels = top.Select(t => Truncate(t.Name, 18)).ToArray(),
                LabelsRotation = 0,
                LabelsPaint = new SolidColorPaint(SKColor.Parse("#9CA3AF")),
                TextSize = 11
            }
        };
        chartClients.YAxes = new Axis[]
        {
            new Axis
            {
                MinLimit = 0,
                LabelsPaint = new SolidColorPaint(SKColor.Parse("#9CA3AF")),
                TextSize = 11
            }
        };
    }

    private void BuildConfidenceChart(List<Document> docs)
    {
        var today = DateTime.Today;
        var days = Enumerable.Range(0, 14).Select(i => today.AddDays(-13 + i)).ToList();

        var values = days.Select(day =>
        {
            var dayDocs = docs.Where(d => d.UploadedAt.Date == day && d.ConfidenceScore.HasValue).ToList();
            return dayDocs.Any() ? dayDocs.Average(d => d.ConfidenceScore!.Value) * 100 : 0;
        }).ToArray();

        chartConfidence.Series = new ISeries[]
        {
            new LineSeries<double>
            {
                Values = values,
                Name = "Confiança média (%)",
                Stroke = new SolidColorPaint(SKColor.Parse("#22C55E")) { StrokeThickness = 2 },
                Fill = new SolidColorPaint(SKColor.Parse("#1A22C55E")),
                GeometrySize = 8,
                GeometryStroke = new SolidColorPaint(SKColor.Parse("#22C55E")) { StrokeThickness = 2 },
                GeometryFill = new SolidColorPaint(SKColors.White)
            }
        };
        chartConfidence.XAxes = new Axis[]
        {
            new Axis
            {
                Labels = days.Select(d => d.ToString("dd/MM")).ToArray(),
                LabelsPaint = new SolidColorPaint(SKColor.Parse("#9CA3AF")),
                TextSize = 10
            }
        };
        chartConfidence.YAxes = new Axis[]
        {
            new Axis
            {
                MinLimit = 0,
                MaxLimit = 100,
                Labeler = v => $"{v:F0}%",
                LabelsPaint = new SolidColorPaint(SKColor.Parse("#9CA3AF")),
                TextSize = 10
            }
        };
    }

    private static SKColor StatusColor(DocumentStatus s) => s switch
    {
        DocumentStatus.Validated => SKColor.Parse("#22C55E"),
        DocumentStatus.ReadyForReview => SKColor.Parse("#F59E0B"),
        DocumentStatus.Pending => SKColor.Parse("#6B7280"),
        DocumentStatus.Processing => SKColor.Parse("#3B82F6"),
        DocumentStatus.Rejected => SKColor.Parse("#EF4444"),
        DocumentStatus.Error => SKColor.Parse("#DC2626"),
        _ => SKColor.Parse("#9CA3AF")
    };

    private static string DisplayName(DocumentStatus s) => s switch
    {
        DocumentStatus.Pending => "Pendente",
        DocumentStatus.Processing => "Processando",
        DocumentStatus.ReadyForReview => "Revisar",
        DocumentStatus.Validated => "Validado",
        DocumentStatus.Rejected => "Rejeitado",
        DocumentStatus.Error => "Erro",
        _ => s.ToString()
    };

    private static string Truncate(string s, int max) => s.Length <= max ? s : s.Substring(0, max - 1) + "…";
}
