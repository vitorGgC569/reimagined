using System.IO;
using System.Text.Json;

namespace OContabil.Services;

/// <summary>
/// Settings persistidos em <c>%LocalAppData%\OContabil\settings.json</c>.
/// Inclui parâmetros do motor de IA, sessão, backup, exportação e telemetria.
/// </summary>
public static class AppSettings
{
    private static readonly string _settingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "OContabil", "settings.json");

    private static SettingsData _data = Load();

    // ── IA / GLiNER ────────────────────────────────────────────────────
    public static string GlinerModelName
    {
        get => _data.GlinerModelName;
        set { _data.GlinerModelName = value; Save(); }
    }

    public static double GlinerThreshold
    {
        get => _data.GlinerThreshold;
        set { _data.GlinerThreshold = value; Save(); }
    }

    public static AiEnginePreference PreferredEngine
    {
        get => _data.PreferredEngine;
        set { _data.PreferredEngine = value; Save(); }
    }

    public static int AiTimeoutSeconds
    {
        get => _data.AiTimeoutSeconds;
        set { _data.AiTimeoutSeconds = Math.Clamp(value, 10, 600); Save(); }
    }

    public static int AiMaxRetries
    {
        get => _data.AiMaxRetries;
        set { _data.AiMaxRetries = Math.Clamp(value, 0, 5); Save(); }
    }

    // ── Sessão ─────────────────────────────────────────────────────────
    public static int SessionTimeoutMinutes
    {
        get => _data.SessionTimeoutMinutes;
        set { _data.SessionTimeoutMinutes = Math.Clamp(value, 1, 240); Save(); }
    }

    // ── Backup ─────────────────────────────────────────────────────────
    public static string BackupFolder
    {
        get => _data.BackupFolder;
        set { _data.BackupFolder = value ?? ""; Save(); }
    }

    public static int BackupMaxFiles
    {
        get => _data.BackupMaxFiles;
        set { _data.BackupMaxFiles = Math.Clamp(value, 1, 365); Save(); }
    }

    public static int BackupIntervalHours
    {
        get => _data.BackupIntervalHours;
        set { _data.BackupIntervalHours = Math.Clamp(value, 1, 168); Save(); }
    }

    // ── Empresa / branding ─────────────────────────────────────────────
    public static string CompanyName
    {
        get => _data.CompanyName;
        set { _data.CompanyName = value ?? ""; Save(); }
    }

    public static string CompanyCnpj
    {
        get => _data.CompanyCnpj;
        set { _data.CompanyCnpj = value ?? ""; Save(); }
    }

    // ── Alertas ────────────────────────────────────────────────────────
    public static int StaleDocumentHours
    {
        get => _data.StaleDocumentHours;
        set { _data.StaleDocumentHours = Math.Clamp(value, 1, 720); Save(); }
    }

    // ── Telemetria ─────────────────────────────────────────────────────
    public static bool TelemetryOptIn
    {
        get => _data.TelemetryOptIn;
        set { _data.TelemetryOptIn = value; Save(); }
    }

    // ── Wizard de primeiro uso ─────────────────────────────────────────
    public static bool FirstRunCompleted
    {
        get => _data.FirstRunCompleted;
        set { _data.FirstRunCompleted = value; Save(); }
    }

    // ── Persistência ───────────────────────────────────────────────────
    private static SettingsData Load()
    {
        try
        {
            if (File.Exists(_settingsPath))
            {
                var json = File.ReadAllText(_settingsPath);
                var data = JsonSerializer.Deserialize<SettingsData>(json) ?? new SettingsData();
                Migrate(data);
                return data;
            }
        }
        catch { }
        return new SettingsData();
    }

    /// <summary>Migra settings legados (preserva escolhas explícitas do usuário).</summary>
    private static void Migrate(SettingsData d)
    {
        // gliner2-base-v1 era apenas o DEFAULT antigo (não há escolha de modelo na UI) e é
        // fraco p/ PT-BR; sobe para o multilíngue (novo default). Modelos custom são mantidos.
        if (string.IsNullOrWhiteSpace(d.GlinerModelName) || d.GlinerModelName == "fastino/gliner2-base-v1")
            d.GlinerModelName = "fastino/gliner2-multi-v1";
    }

    private static void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_settingsPath)!);
            var json = JsonSerializer.Serialize(_data, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(_settingsPath, json);
        }
        catch { }
    }

    private sealed class SettingsData
    {
        public string GlinerModelName { get; set; } = "fastino/gliner2-multi-v1";
        public double GlinerThreshold { get; set; } = 0.80;
        public AiEnginePreference PreferredEngine { get; set; } = AiEnginePreference.Auto;
        public int AiTimeoutSeconds { get; set; } = 120;
        public int AiMaxRetries { get; set; } = 1;
        public int SessionTimeoutMinutes { get; set; } = 20;
        public string BackupFolder { get; set; } = "";
        public int BackupMaxFiles { get; set; } = 30;
        public int BackupIntervalHours { get; set; } = 24;
        public string CompanyName { get; set; } = "OContabil";
        public string CompanyCnpj { get; set; } = "";
        public int StaleDocumentHours { get; set; } = 24;
        public bool TelemetryOptIn { get; set; }
        public bool FirstRunCompleted { get; set; }
    }
}

public enum AiEnginePreference
{
    Auto = 0,
    Onnx = 1,
    Python = 2,
    Regex = 3
}
