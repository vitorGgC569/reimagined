using System.IO;
using System.Windows;
using System.Windows.Threading;
using System.Threading.Tasks;
using OContabil.Services;
using Serilog;
using Serilog.Events;

namespace OContabil;

public partial class App : Application
{
    private static readonly string _logFolder = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "OContabil", "logs");

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        ConfigureSerilog();

        Log.Information("OContabil iniciado — versão {Version}",
            System.Reflection.Assembly.GetExecutingAssembly().GetName().Version);

        // Catch UI thread exceptions
        this.DispatcherUnhandledException += App_DispatcherUnhandledException;
        // Catch background thread exceptions
        AppDomain.CurrentDomain.UnhandledException += CurrentDomain_UnhandledException;
        // Catch unobserved Task exceptions
        TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Log.Information("OContabil encerrando");
        DocumentProcessingQueue.Instance.Shutdown();
        Log.CloseAndFlush();
        base.OnExit(e);
    }

    private static void ConfigureSerilog()
    {
        try
        {
            Directory.CreateDirectory(_logFolder);
            var logPath = Path.Combine(_logFolder, "ocontabil-.log");
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .MinimumLevel.Override("Microsoft", LogEventLevel.Warning)
                .Enrich.WithProperty("App", "OContabil")
                .WriteTo.File(logPath,
                    rollingInterval: RollingInterval.Day,
                    retainedFileCountLimit: 14,
                    outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
                .CreateLogger();
        }
        catch
        {
            // Se não conseguir criar logger, segue sem — falha silenciosa.
        }
    }

    private void App_DispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        LogCrash(e.Exception, "UI Thread");
        ShowFriendlyError(e.Exception);
        e.Handled = true;
    }

    private void CurrentDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        if (e.ExceptionObject is Exception ex)
        {
            LogCrash(ex, "Background Thread");
        }
    }

    private void TaskScheduler_UnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        LogCrash(e.Exception, "Async Task");
        e.SetObserved();
    }

    private static void LogCrash(Exception ex, string source)
    {
        try
        {
            Log.Error(ex, "Crash em {Source}: {Message}", source, ex.Message);
        }
        catch { }
    }

    private static void ShowFriendlyError(Exception ex)
    {
        var friendly = MapMessage(ex);
        MessageBox.Show(
            friendly +
            $"\n\nUm registro detalhado foi gravado em logs/ ({_logFolder}).",
            "OContabil", MessageBoxButton.OK, MessageBoxImage.Warning);
    }

    private static string MapMessage(Exception ex)
    {
        return ex switch
        {
            UnauthorizedAccessException => "Permissão negada ao acessar o arquivo. Verifique se ele não está aberto em outro programa.",
            FileNotFoundException => "Arquivo não encontrado. O documento pode ter sido movido ou excluído.",
            DirectoryNotFoundException => "Pasta não encontrada. Verifique se o caminho configurado ainda existe.",
            IOException => "Erro de leitura/escrita em disco. Pode haver falta de espaço ou bloqueio por antivírus.",
            System.Net.Http.HttpRequestException => "Falha de rede. Verifique sua conexão e tente novamente.",
            TimeoutException => "Tempo esgotado. A operação demorou mais do que o limite configurado.",
            _ => "Ocorreu um erro inesperado. Um registro técnico (sem dados sensíveis) foi gravado nos logs."
        };
    }
}
