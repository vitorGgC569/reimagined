using System.Windows;

namespace OContabil.Services;

public static class ThemeManager
{
    public static string Current { get; private set; } = "Dark";

    public static void ApplyTheme(string themeName)
    {
        var app = Application.Current;
        if (app == null) return;

        // Try to locate existing theme dict
        var existing = app.Resources.MergedDictionaries.FirstOrDefault(d => 
            d.Source != null && d.Source.OriginalString.Contains("Theme.xaml"));

        var newThemeSource = new Uri($"pack://application:,,,/OContabil;component/Themes/{themeName}Theme.xaml", UriKind.Absolute);
        
        // Use relative for local testing if needed:
        if (!Uri.TryCreate($"pack://application:,,,/OContabil;component/Themes/{themeName}Theme.xaml", UriKind.Absolute, out _))
        {
            newThemeSource = new Uri($"Themes/{themeName}Theme.xaml", UriKind.Relative);
        }

        var newTheme = new ResourceDictionary { Source = newThemeSource };

        if (existing != null)
        {
            app.Resources.MergedDictionaries.Remove(existing);
        }
        
        app.Resources.MergedDictionaries.Add(newTheme);
        Current = themeName;
    }

    public static void ToggleTheme()
    {
        ApplyTheme(Current == "Dark" ? "Light" : "Dark");
    }
}
