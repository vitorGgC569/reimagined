using System.IO;
using System.Windows.Threading;
using OContabil.Data;

namespace OContabil.Services;

/// <summary>
/// Faz cópias diárias do arquivo SQLite (ocontabil.db) para uma pasta de backup
/// configurável e mantém apenas os últimos N arquivos. Executa em background
/// usando <see cref="DispatcherTimer"/>.
/// </summary>
public sealed class BackupService
{
    private static readonly Lazy<BackupService> _shared = new(() => new BackupService());
    public static BackupService Instance => _shared.Value;

    private readonly DispatcherTimer _timer;
    private string _backupFolder = string.Empty;
    private int _maxBackups = 30;
    private TimeSpan _interval = TimeSpan.FromHours(24);
    private DateTime _lastBackup = DateTime.MinValue;

    public string BackupFolder => _backupFolder;
    public DateTime LastBackup => _lastBackup;

    private BackupService()
    {
        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromMinutes(5)
        };
        _timer.Tick += (_, _) => RunIfDue();
    }

    public void ConfigureFromSettings()
    {
        Configure(AppSettings.BackupFolder, AppSettings.BackupMaxFiles, AppSettings.BackupIntervalHours);
    }

    public void Configure(string folder, int maxBackups, int intervalHours)
    {
        if (string.IsNullOrWhiteSpace(folder))
        {
            folder = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "OContabil", "backups");
        }
        Directory.CreateDirectory(folder);

        _backupFolder = folder;
        _maxBackups = Math.Max(1, maxBackups);
        _interval = TimeSpan.FromHours(Math.Max(1, intervalHours));
    }

    public void Start()
    {
        if (!_timer.IsEnabled) _timer.Start();
        RunIfDue();
    }

    public void Stop()
    {
        if (_timer.IsEnabled) _timer.Stop();
    }

    public string? RunNow()
    {
        try
        {
            if (string.IsNullOrEmpty(_backupFolder))
                ConfigureFromSettings();

            var dbPath = AppDbContext.DatabasePath;
            if (!File.Exists(dbPath)) return null;

            var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var dest = Path.Combine(_backupFolder, $"ocontabil_{stamp}.db");
            File.Copy(dbPath, dest, overwrite: false);
            _lastBackup = DateTime.UtcNow;

            CleanupOldBackups();
            return dest;
        }
        catch
        {
            return null;
        }
    }

    private void RunIfDue()
    {
        if (DateTime.UtcNow - _lastBackup < _interval) return;
        RunNow();
    }

    private void CleanupOldBackups()
    {
        try
        {
            var files = new DirectoryInfo(_backupFolder)
                .GetFiles("ocontabil_*.db")
                .OrderByDescending(f => f.CreationTimeUtc)
                .ToArray();

            for (int i = _maxBackups; i < files.Length; i++)
            {
                try { files[i].Delete(); } catch { }
            }
        }
        catch { }
    }
}
