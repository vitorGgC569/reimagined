using System.Diagnostics;
using System.IO;
using System.Text.Json;

namespace OContabil.Services;

/// <summary>
/// Wrapper sobre o script Python <c>gliner_bridge.py</c>. Lança o interpretador
/// como subprocesso, repassa argumentos do schema/threshold e devolve um
/// <see cref="GlinerResult"/> tipado.
/// </summary>
public class GlinerService
{
    private readonly string _scriptPath;
    private readonly string _pythonPath;

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
        string[] candidates =
        {
            Environment.GetEnvironmentVariable("OCONTABIL_PYTHON") ?? "",
            "python",
            "python3",
            "py"
        };

        foreach (var cmd in candidates.Where(c => !string.IsNullOrEmpty(c)))
        {
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
        {
            return new GlinerResult
            {
                Success = false,
                Error = "Python não encontrado. Instale Python 3.10+ e execute pip install -r Scripts/requirements.txt"
            };
        }
        if (!IsScriptAvailable)
        {
            return new GlinerResult { Success = false, Error = $"Script não encontrado: {_scriptPath}" };
        }

        try
        {
            var modelName = AppSettings.GlinerModelName;
            var threshold = AppSettings.GlinerThreshold;

            var psi = new ProcessStartInfo(_pythonPath)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,             // nunca via shell (anti command-injection)
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(_scriptPath) ?? ""
            };
            // Argumentos como LISTA — cada um escapado pelo runtime. Elimina
            // arg/command-injection via aspas/metacaracteres no caminho do arquivo.
            psi.ArgumentList.Add(_scriptPath);
            psi.ArgumentList.Add(filePath);
            psi.ArgumentList.Add(docType ?? "");
            psi.ArgumentList.Add(modelName ?? "");
            psi.ArgumentList.Add(threshold.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
            psi.Environment["PYTHONIOENCODING"] = "utf-8";

            using var process = Process.Start(psi)!;

            var stdoutTask = process.StandardOutput.ReadToEndAsync(ct);
            var stderrTask = process.StandardError.ReadToEndAsync(ct);

            try
            {
                await process.WaitForExitAsync(ct);
            }
            catch (OperationCanceledException)
            {
                // Resiliência/sandbox: mata o processo (e filhos) p/ não deixar órfão.
                try { if (!process.HasExited) process.Kill(entireProcessTree: true); } catch { }
                throw;
            }

            var stdout = await stdoutTask;
            var stderr = await stderrTask;

            int jsonStart = stdout.IndexOf('{');
            int jsonEnd = stdout.LastIndexOf('}');
            if (jsonStart >= 0 && jsonEnd > jsonStart)
                stdout = stdout.Substring(jsonStart, jsonEnd - jsonStart + 1);

            if (string.IsNullOrWhiteSpace(stdout) || !stdout.StartsWith("{"))
            {
                SafeLog.Warn("gliner.bridge", $"exit={process.ExitCode}; stderr={stderr}");
                return new GlinerResult
                {
                    Success = false,
                    Error = "O extrator Python não retornou um resultado válido."
                };
            }

            var result = JsonSerializer.Deserialize<GlinerResult>(stdout, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
            return result ?? new GlinerResult { Success = false, Error = "Falha ao desserializar resultado." };
        }
        catch (OperationCanceledException)
        {
            return new GlinerResult
            {
                Success = false,
                Error = "Processamento cancelado por timeout."
            };
        }
        catch (Exception ex)
        {
            SafeLog.Error("gliner.process", ex);
            return new GlinerResult { Success = false, Error = "Falha no extrator Python." };
        }
    }
}
