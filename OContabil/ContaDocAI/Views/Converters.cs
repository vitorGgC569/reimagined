using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace ContaDocAI.Views;

public static class Converters
{
    public static IValueConverter VisibleIfPositive { get; } = new VisibleIfPositiveConverter();
    public static IValueConverter ConfidenceToColor { get; } = new ConfidenceToColorConverter();
    public static IValueConverter StringToColor { get; } = new StringToColorConverter();
}

public class VisibleIfPositiveConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is int i) return i > 0 ? Visibility.Visible : Visibility.Collapsed;
        return Visibility.Collapsed;
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}

public class ConfidenceToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d)
        {
            if (d >= 0.9) return new SolidColorBrush(Color.FromRgb(16, 185, 129));   // #10b981
            if (d >= 0.8) return new SolidColorBrush(Color.FromRgb(245, 158, 11));   // #f59e0b
            return new SolidColorBrush(Color.FromRgb(239, 68, 68));                  // #ef4444
        }
        return new SolidColorBrush(Colors.Gray);
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}

public class PercentToWidthConverter : IMultiValueConverter
{
    public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
    {
        if (values.Length == 2 && values[0] is int percent && values[1] is double maxWidth)
            return percent / 100.0 * maxWidth;
        return 0.0;
    }
    public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture) => throw new NotImplementedException();
}

public class StringToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string hex)
        {
            try
            {
                var c = (Color)ColorConverter.ConvertFromString(hex);
                return new SolidColorBrush(c);
            }
            catch { }
        }
        return new SolidColorBrush(Color.FromRgb(99, 102, 241));
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}
