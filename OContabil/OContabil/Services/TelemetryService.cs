using System.IO;
using System.Text.Json;

namespace OContabil.Services;

/// <summary>
/// Telemetria estritamente local (opt-in). Mantém contadores agregados em
/// <c>%LocalAppData%\OContabil\telemetry.json</c> — sem identificadores pessoais,
/// IPs, conteúdo de documentos ou qualquer dado que saia da máquina.
/// Honrar o flag <see cref="AppSettings.TelemetryOptIn"/> antes de incrementar.
/// </summary>
public static class TelemetryService
{
    private static readonly string _path = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "OContabil", "telemetry.json");

    private static readonly object _lock = new();

    public sealed class Counters
    {
        public long DocumentsProcessed { get; set; }
        public long DocumentsValidated { get; set; }
        public long AiFailures { get; set; }
        public long ExportsGenerated { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    public static void IncrementProcessed() => Increment(c => c.DocumentsProcessed++);
    public static void IncrementValidated() => Increment(c => c.DocumentsValidated++);
    public static void IncrementAiFailure() => Increment(c => c.AiFailures++);
    public static void IncrementExport() => Increment(c => c.ExportsGenerated++);

    public static Counters Snapshot()
    {
        lock (_lock) return Load();
    }

    public static void Reset()
    {
        lock (_lock) Save(new Counters { LastUpdated = DateTime.Now });
    }

    private static void Increment(Action<Counters> mutator)
    {
        if (!AppSettings.TelemetryOptIn) return;
        lock (_lock)
        {
            var c = Load();
            mutator(c);
            c.LastUpdated = DateTime.Now;
            Save(c);
        }
    }

    private static Counters Load()
    {
        try
        {
            if (File.Exists(_path))
            {
                var json = File.ReadAllText(_path);
                return JsonSerializer.Deserialize<Counters>(json) ?? new Counters();
            }
        }
        catch { }
        return new Counters();
    }

    private static void Save(Counters c)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            File.WriteAllText(_path, JsonSerializer.Serialize(c, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { }
    }
}
