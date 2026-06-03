using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace OContabil.Services;

/// <summary>
/// Sidecar PERSISTENTE do GLiNER2. Mantém um processo Python (gliner_bridge.py --serve)
/// vivo: o modelo é carregado UMA vez e cada documento é uma requisição (linha JSON via
/// stdin/stdout). Isso elimina o custo de recarregar o modelo por documento (~dezenas de
/// segundos → &lt;1s por doc após a carga inicial). O processo é reiniciado se morrer ou
/// travar (timeout). Acesso serializado por um SemaphoreSlim.
/// </summary>
public sealed class GlinerService : IDisposable
{
    private static readonly Encoding Utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

    private readonly string _scriptPath;
    private readonly string _pythonPath;
    private readonly SemaphoreSlim _gate = new(1, 1);

    private Process? _proc;
    private StreamWriter? _stdin;
    private StreamReader? _stdout;
    private string _modelName = "";

    public GlinerService()
    {
        _scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "gliner_bridge.py");
        if (!File.Exists(_scriptPath))
        {
            var devPath = Path.Combine(Directory.GetCurrentDirectory(), "Scripts", "gliner_bridge.py");
            if (File.Exists(devPath)) _scriptPath = devPath;
        }
        _pythonPath = FindPython();
    }

    public bool IsPythonAvailable => !string.IsNullOrEmpty(_pythonPath);
    public bool IsScriptAvailable => File.Exists(_scriptPath);

    private static string FindPython()
    {
        var bundled = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "python.exe");
        string[] candidates =
        {
            Environment.GetEnvironmentVariable("OCONTABIL_PYTHON") ?? "",
            File.Exists(bundled) ? bundled : "",   // Python embarcado pelo instalador (offline)
            "python",
            "python3",
            "py"
        };
        foreach (var cmd in candidates)
        {
            if (string.IsNullOrEmpty(cmd)) continue;
            try
            {
                var psi = new ProcessStartInfo(cmd, "--version")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                var p = Process.Start(psi);
                p?.WaitForExit(3000);
                if (p?.ExitCode == 0) return cmd;
            }
            catch { }
        }
        return "";
    }

    public async Task<GlinerResult> ProcessFileAsync(string filePath, string docType = "", CancellationToken ct = default)
    {
        if (!IsPythonAvailable)
            return Fail("Python não encontrado. Instale Python 3.10+ e gliner2, ou use o motor determinístico.");
        if (!IsScriptAvailable)
            return Fail($"Script não encontrado: {_scriptPath}");

        await _gate.WaitAsync(ct);
        try
        {
            await EnsureStartedAsync(ct);
            if (_stdin == null || _stdout == null || _proc == null || _proc.HasExited)
                return Fail("Sidecar GLiNER indisponível.");

            // Argumentos via JSON (sem shell): caminho do arquivo escapado pelo serializer.
            var req = JsonSerializer.Serialize(new
            {
                file = filePath,
                doc_type = docType ?? "",
                threshold = AppSettings.GlinerThreshold
            });
            await _stdin.WriteLineAsync(req);
            await _stdin.FlushAsync();

            using var respCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            respCts.CancelAfter(TimeSpan.FromSeconds(Math.Max(20, AppSettings.AiTimeoutSeconds)));
            string? line;
            try { line = await _stdout.ReadLineAsync(respCts.Token); }
            catch (OperationCanceledException) { KillProc(); return Fail("Timeout aguardando o sidecar GLiNER."); }

            if (line == null) { KillProc(); return Fail("Sidecar GLiNER encerrou inesperadamente."); }

            var result = JsonSerializer.Deserialize<GlinerResult>(line,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            return result ?? Fail("Resposta inválida do sidecar GLiNER.");
        }
        catch (OperationCanceledException)
        {
            KillProc();
            return Fail("Processamento GLiNER cancelado.");
        }
        catch (Exception ex)
        {
            SafeLog.Error("gliner.sidecar", ex);
            KillProc();
            return Fail("Falha no sidecar GLiNER.");
        }
        finally { _gate.Release(); }
    }

    private async Task EnsureStartedAsync(CancellationToken ct)
    {
        if (_proc != null && !_proc.HasExited && _stdin != null && _stdout != null) return;
        KillProc();

        _modelName = AppSettings.GlinerModelName ?? "fastino/gliner2-multi-v1";
        var psi = new ProcessStartInfo(_pythonPath)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Utf8NoBom,
            StandardInputEncoding = Utf8NoBom,
            WorkingDirectory = Path.GetDirectoryName(_scriptPath) ?? ""
        };
        psi.ArgumentList.Add(_scriptPath);
        psi.ArgumentList.Add("--serve");
        psi.ArgumentList.Add(_modelName);
        psi.ArgumentList.Add(AppSettings.GlinerThreshold.ToString("F2", CultureInfo.InvariantCulture));
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        psi.Environment["HF_HUB_DISABLE_TELEMETRY"] = "1";
        // Modelo GLiNER embutido pelo instalador (models\hf) => roda 100% offline.
        var hfDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "hf");
        if (Directory.Exists(hfDir))
        {
            psi.Environment["HF_HOME"] = hfDir;
            psi.Environment["HF_HUB_OFFLINE"] = "1";
            psi.Environment["TRANSFORMERS_OFFLINE"] = "1";
        }

        _proc = Process.Start(psi)!;
        _stdin = _proc.StandardInput;
        _stdin.AutoFlush = true;
        _stdout = _proc.StandardOutput;
        // Drena stderr (logs de carga/inferência) p/ não bloquear o buffer do processo.
        _ = Task.Run(async () => { try { await _proc.StandardError.ReadToEndAsync(); } catch { } });

        // Aguarda a linha "ready" — a carga do modelo pode levar dezenas de segundos.
        using var startCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        startCts.CancelAfter(TimeSpan.FromSeconds(Math.Max(180, AppSettings.AiTimeoutSeconds)));
        string? ready;
        try { ready = await _stdout.ReadLineAsync(startCts.Token); }
        catch (OperationCanceledException) { KillProc(); throw new TimeoutException("Sidecar GLiNER não inicializou a tempo."); }
        if (ready == null) { KillProc(); throw new InvalidOperationException("Sidecar GLiNER não respondeu na inicialização."); }
    }

    private void KillProc()
    {
        try { if (_proc != null && !_proc.HasExited) _proc.Kill(entireProcessTree: true); } catch { }
        try { _stdin?.Dispose(); } catch { }
        _proc = null; _stdin = null; _stdout = null;
    }

    private static GlinerResult Fail(string error) => new() { Success = false, Error = error };

    public void Dispose()
    {
        try
        {
            if (_proc != null && !_proc.HasExited && _stdin != null)
            {
                try { _stdin.WriteLine("{\"cmd\":\"quit\"}"); _stdin.Flush(); } catch { }
            }
        }
        catch { }
        KillProc();
        _gate.Dispose();
    }
}
