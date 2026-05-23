using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using ContaDocAI.Models;
using ContaDocAI.Services;

namespace ContaDocAI.Views;

public partial class ValidationView : UserControl
{
    private List<Document> _docs = new();
    private int _currentIndex;

    public ValidationView()
    {
        InitializeComponent();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        _docs = MockDataService.ValidationQueue;
        _currentIndex = 0;
        RenderDocument();
    }

    private void RenderDocument()
    {
        if (_docs.Count == 0) return;

        var doc = _docs[_currentIndex];
        txtPosition.Text = $"{_currentIndex + 1} de {_docs.Count}";
        txtDocName.Text = doc.Filename;
        txtDocType.Text = doc.DocumentType;
        btnPrev.IsEnabled = _currentIndex > 0;
        btnNext.IsEnabled = _currentIndex < _docs.Count - 1;

        // Average confidence
        var avgConf = doc.ExtractedFields.Values.Average(f => f.Confidence);
        var avgPercent = (int)(avgConf * 100);
        txtAvgConf.Text = $"GLiNER 2 · Confianca media: {avgPercent}%";
        txtOverallConf.Text = $"{avgPercent}%";
        txtOverallConf.Foreground = avgConf >= 0.9
            ? (Brush)FindResource("SuccessBrush")
            : avgConf >= 0.8
                ? (Brush)FindResource("WarningBrush")
                : (Brush)FindResource("ErrorBrush");

        // OCR Text with highlights
        RenderOcrText(doc);

        // Form fields
        RenderFields(doc);

        // Doc chips
        RenderChips();
    }

    private void RenderOcrText(Document doc)
    {
        txtOcrContent.Inlines.Clear();
        var text = doc.OcrText;

        // Collect spans
        var highlights = doc.ExtractedFields
            .Where(f => f.Value.SpanStart >= 0 && f.Value.SpanEnd <= text.Length && f.Value.SpanStart < f.Value.SpanEnd)
            .Select(f => new { Field = f.Key, f.Value.SpanStart, f.Value.SpanEnd, f.Value.Confidence })
            .OrderBy(h => h.SpanStart)
            .ToList();

        int lastEnd = 0;
        foreach (var h in highlights)
        {
            if (h.SpanStart < lastEnd) continue;

            // Text before highlight
            if (h.SpanStart > lastEnd)
            {
                txtOcrContent.Inlines.Add(new Run(text[lastEnd..h.SpanStart]));
            }

            // Highlighted span
            var highlightColor = h.Confidence >= 0.9
                ? Color.FromRgb(16, 185, 129)
                : h.Confidence >= 0.8
                    ? Color.FromRgb(245, 158, 11)
                    : Color.FromRgb(239, 68, 68);

            var run = new Run(text[h.SpanStart..h.SpanEnd])
            {
                Background = new SolidColorBrush(Color.FromArgb(0x33, highlightColor.R, highlightColor.G, highlightColor.B)),
                Foreground = new SolidColorBrush(highlightColor),
                FontWeight = FontWeights.SemiBold,
                ToolTip = $"{h.Field}: {(int)(h.Confidence * 100)}%"
            };

            var underline = new Underline(run);
            txtOcrContent.Inlines.Add(underline);

            lastEnd = h.SpanEnd;
        }

        // Remaining text
        if (lastEnd < text.Length)
        {
            txtOcrContent.Inlines.Add(new Run(text[lastEnd..]));
        }
    }

    private void RenderFields(Document doc)
    {
        fieldsPanel.Children.Clear();

        foreach (var (fieldName, data) in doc.ExtractedFields)
        {
            var confPercent = (int)(data.Confidence * 100);
            var confColor = data.Confidence >= 0.9
                ? Color.FromRgb(16, 185, 129)
                : data.Confidence >= 0.8
                    ? Color.FromRgb(245, 158, 11)
                    : Color.FromRgb(239, 68, 68);

            var fieldGroup = new StackPanel { Margin = new Thickness(0, 0, 0, 14) };

            // Label + confidence row
            var labelRow = new DockPanel { Margin = new Thickness(0, 0, 0, 4) };
            labelRow.Children.Add(new TextBlock
            {
                Text = fieldName.Replace("_", " "),
                FontSize = 12,
                FontWeight = FontWeights.SemiBold,
                Foreground = (Brush)FindResource("TextSecondaryBrush")
            });

            // Confidence meter
            var confPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right
            };
            DockPanel.SetDock(confPanel, Dock.Right);

            var confTrack = new Border
            {
                Width = 60, Height = 4, CornerRadius = new CornerRadius(2),
                Background = (Brush)FindResource("ElevatedBrush"),
                Margin = new Thickness(0, 0, 6, 0),
                VerticalAlignment = VerticalAlignment.Center
            };
            var confFill = new Border
            {
                Width = confPercent * 0.6, Height = 4, CornerRadius = new CornerRadius(2),
                Background = new SolidColorBrush(confColor),
                HorizontalAlignment = HorizontalAlignment.Left
            };
            confTrack.Child = confFill;
            confPanel.Children.Add(confTrack);
            confPanel.Children.Add(new TextBlock
            {
                Text = $"{confPercent}%",
                FontSize = 10,
                FontWeight = FontWeights.Bold,
                Foreground = new SolidColorBrush(confColor),
                VerticalAlignment = VerticalAlignment.Center
            });

            labelRow.Children.Add(confPanel);
            fieldGroup.Children.Add(labelRow);

            // Input
            var input = new TextBox
            {
                Text = data.Value,
                Style = (Style)FindResource("DarkTextBox"),
                Tag = fieldName
            };

            // Low confidence warning border
            if (data.Confidence < 0.8)
            {
                input.BorderBrush = new SolidColorBrush(Color.FromArgb(0x66, 0xef, 0x44, 0x44));
            }

            fieldGroup.Children.Add(input);

            // Warning hint for low confidence
            if (confPercent < 80)
            {
                fieldGroup.Children.Add(new TextBlock
                {
                    Text = "⚠️ Confianca baixa — revise manualmente",
                    FontSize = 10,
                    Foreground = (Brush)FindResource("WarningBrush"),
                    Margin = new Thickness(0, 4, 0, 0)
                });
            }

            fieldsPanel.Children.Add(fieldGroup);
        }
    }

    private void RenderChips()
    {
        docChips.Children.Clear();
        for (int i = 0; i < _docs.Count; i++)
        {
            var idx = i;
            var btn = new Button
            {
                Content = $"{GetDocIcon(_docs[i].DocumentType)} {_docs[i].Id}",
                Style = i == _currentIndex
                    ? (Style)FindResource("BtnPrimary")
                    : (Style)FindResource("BtnSecondary"),
                Padding = new Thickness(10, 5, 10, 5),
                FontSize = 11,
                Margin = new Thickness(0, 0, 6, 0)
            };
            btn.Click += (_, _) => { _currentIndex = idx; RenderDocument(); };
            docChips.Children.Add(btn);
        }
    }

    private string GetDocIcon(string type) => type switch
    {
        "Nota Fiscal" => "📃",
        "Boleto" => "🏦",
        "Comprovante PIX" => "💸",
        _ => "📄"
    };

    private void OnPrevClick(object sender, RoutedEventArgs e)
    {
        if (_currentIndex > 0) { _currentIndex--; RenderDocument(); }
    }

    private void OnNextClick(object sender, RoutedEventArgs e)
    {
        if (_currentIndex < _docs.Count - 1) { _currentIndex++; RenderDocument(); }
    }

    private void OnValidate(object sender, RoutedEventArgs e)
    {
        var doc = _docs[_currentIndex];
        MessageBox.Show($"Documento {doc.Id} validado com sucesso!", "ContaDoc AI",
            MessageBoxButton.OK, MessageBoxImage.Information);

        _docs.RemoveAt(_currentIndex);
        if (_currentIndex >= _docs.Count) _currentIndex = Math.Max(0, _docs.Count - 1);

        if (_docs.Count > 0)
            RenderDocument();
        else
            MessageBox.Show("Todos os documentos foram validados!", "ContaDoc AI",
                MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private void OnReject(object sender, RoutedEventArgs e)
    {
        OnNextClick(sender, e);
    }

    private void OnSkip(object sender, RoutedEventArgs e)
    {
        OnNextClick(sender, e);
    }
}
