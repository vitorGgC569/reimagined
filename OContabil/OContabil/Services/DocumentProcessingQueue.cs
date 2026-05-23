using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows;
using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services;

/// <summary>
/// Fila assíncrona de processamento de documentos. Aplica a cadeia configurável
/// de motores (ONNX → Python → Regex), com timeout, retries, validação cruzada
/// e classificação automática.
/// </summary>
public sealed class DocumentProcessingQueue
{
    private static DocumentProcessingQueue? _instance;
    public static DocumentProcessingQueue Instance => _instance ??= new DocumentProcessingQueue();

    private readonly ConcurrentQueue<QueueItem> _queue = new();
    private readonly SemaphoreSlim _signal = new(0);
    private readonly GlinerService _gliner = new();
    private readonly GlinerOnnxService _onnx = new();
    private readonly CancellationTokenSource _cts = new();
    private bool _isRunning;
    private bool _isNativeReady;

    public int PendingCount => _queue.Count;

    public event Action<int>? DocumentProcessed;

    private DocumentProcessingQueue()
    {
        try
        {
            if (_onnx.IsModelAvailable)
            {
                _onnx.LoadModel();
                _isNativeReady = _onnx.IsLoaded;
            }
        }
        catch { _isNativeReady = false; }

        StartConsumer();
    }

    public void Enqueue(int docId, string filePath, string docType)
    {
        try
        {
            using var db = new AppDbContext();
            var doc = db.Documents.Find(docId);
            if (doc != null)
            {
                doc.Status = DocumentStatus.Processing;
                doc.OcrText = null;
                doc.ProcessedAt = null;
                db.SaveChanges();
            }
        }
        catch { }

        _queue.Enqueue(new QueueItem(docId, filePath, docType));
        _signal.Release();

        Application.Current?.Dispatcher.Invoke(() =>
            ToastService.ShowInfo($"Documento enfileirado para processamento ({_queue.Count} na fila)."));
    }

    private void StartConsumer()
    {
        if (_isRunning) return;
        _isRunning = true;

        Task.Run(async () =>
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                await _signal.WaitAsync(_cts.Token);
                if (_queue.TryDequeue(out var item))
                {
                    await ProcessItemAsync(item);
                    Application.Current?.Dispatcher.Invoke(() =>
                        DocumentProcessed?.Invoke(item.DocId));
                }
            }
        }, _cts.Token);
    }

    private async Task ProcessItemAsync(QueueItem item)
    {
        try
        {
            var result = await RunPipelineAsync(item);
            ApplyResult(item, result);
        }
        catch (Exception ex)
        {
            MarkAsError(item.DocId, $"Erro inesperado: {ex.Message}");
        }
    }

    private async Task<GlinerResult> RunPipelineAsync(QueueItem item)
    {
        var threshold = AppSettings.GlinerThreshold;
        var preference = AppSettings.PreferredEngine;
        var timeout = TimeSpan.FromSeconds(AppSettings.AiTimeoutSeconds);
        var maxRetries = AppSettings.AiMaxRetries;

        var order = BuildEngineOrder(preference, item.FilePath);

        Exception? lastError = null;
        foreach (var engine in order)
        {
            for (int attempt = 0; attempt <= maxRetries; attempt++)
            {
                try
                {
                    using var cts = new CancellationTokenSource(timeout);
                    var task = engine switch
                    {
                        AiEnginePreference.Onnx => RunOnnxAsync(item, threshold, cts.Token),
                        AiEnginePreference.Python => RunPythonAsync(item, cts.Token),
                        _ => RunRegexAsync(item, threshold)
                    };
                    var result = await task;
                    if (result.Success)
                    {
                        result.Note = $"engine={engine}; attempt={attempt + 1}";
                        return result;
                    }
                    lastError = new Exception(result.Error ?? "Falha sem detalhe.");
                }
                catch (OperationCanceledException)
                {
                    lastError = new TimeoutException($"Timeout no motor {engine} ({timeout.TotalSeconds:F0}s).");
                }
                catch (Exception ex)
                {
                    lastError = ex;
                }
            }
        }

        return new GlinerResult
        {
            Success = false,
            Error = $"Todos os motores falharam. Ultimo erro: {lastError?.Message ?? "desconhecido"}"
        };
    }

    private static List<AiEnginePreference> BuildEngineOrder(AiEnginePreference preference, string filePath)
    {
        var onnxOk = Instance._isNativeReady && DocumentTextExtractor.CanReadNatively(filePath);
        return preference switch
        {
            AiEnginePreference.Onnx => onnxOk
                ? new() { AiEnginePreference.Onnx, AiEnginePreference.Python, AiEnginePreference.Regex }
                : new() { AiEnginePreference.Python, AiEnginePreference.Regex },
            AiEnginePreference.Python => new() { AiEnginePreference.Python, AiEnginePreference.Regex },
            AiEnginePreference.Regex => new() { AiEnginePreference.Regex },
            _ => onnxOk
                ? new() { AiEnginePreference.Onnx, AiEnginePreference.Python, AiEnginePreference.Regex }
                : new() { AiEnginePreference.Python, AiEnginePreference.Regex }
        };
    }

    private async Task<GlinerResult> RunOnnxAsync(QueueItem item, double threshold, CancellationToken ct)
    {
        var text = await DocumentTextExtractor.ReadAsync(item.FilePath, ct);
        if (string.IsNullOrWhiteSpace(text))
            return new GlinerResult { Success = false, Error = "Sem texto extraível para ONNX." };

        var labels = GetLabelsForType(item.DocType);
        return await _onnx.PredictAsync(text, labels, (float)threshold);
    }

    private async Task<GlinerResult> RunPythonAsync(QueueItem item, CancellationToken ct)
    {
        if (!_gliner.IsPythonAvailable)
            return new GlinerResult { Success = false, Error = "Python não disponível." };
        return await _gliner.ProcessFileAsync(item.FilePath, item.DocType, ct);
    }

    private async Task<GlinerResult> RunRegexAsync(QueueItem item, double threshold)
    {
        var text = await DocumentTextExtractor.ReadAsync(item.FilePath);
        if (string.IsNullOrWhiteSpace(text))
        {
            // Tenta extrair texto pelo Python (PDF/OCR) e em seguida regex.
            try
            {
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(AppSettings.AiTimeoutSeconds));
                var pyResult = await _gliner.ProcessFileAsync(item.FilePath, item.DocType, cts.Token);
                text = pyResult.OcrText ?? "";
            }
            catch { }
        }
        return RegexExtractionService.Extract(text ?? "", item.DocType, threshold);
    }

    private void ApplyResult(QueueItem item, GlinerResult result)
    {
        using var db = new AppDbContext();
        var doc = db.Documents.Include(d => d.Client).FirstOrDefault(d => d.Id == item.DocId);
        if (doc == null) return;

        doc.ProcessedAt = DateTime.Now;

        if (!result.Success)
        {
            doc.Status = DocumentStatus.Error;
            doc.OcrText = result.Error ?? "Erro desconhecido.";
            db.SaveChanges();
            AuditLogger.Write(db, null, "document.process.error", "Documents", doc.Id, result.Error);
            TelemetryService.IncrementAiFailure();
            return;
        }

        var confidence = result.AvgConfidence > 0
            ? result.AvgConfidence
            : InferConfidenceFromKeyCount(result);

        var jsonString = result.Extraction?.ToString();
        doc.OcrText = result.OcrText;
        doc.ExtractedJson = jsonString;
        doc.ExtractedJsonOriginal = jsonString;

        var threshold = AppSettings.GlinerThreshold;
        bool crossValidated = CrossValidate(doc, result, db);
        bool meets = confidence >= threshold;

        if (crossValidated)
        {
            doc.Status = DocumentStatus.Validated;
            doc.ConfidenceScore = 1.0;
            doc.ValidatedAt = DateTime.Now;
        }
        else if (meets)
        {
            doc.Status = DocumentStatus.Validated;
            doc.ConfidenceScore = confidence;
            doc.ValidatedAt = DateTime.Now;
        }
        else
        {
            doc.Status = DocumentStatus.ReadyForReview;
            doc.ConfidenceScore = confidence;
        }

        AutoCategorize(doc, result, db);

        if (doc.Status == DocumentStatus.Validated && doc.Client != null)
            doc.Client.ValidatedCount++;

        db.SaveChanges();
        AuditLogger.Write(db, null, "document.processed", "Documents", doc.Id,
            $"status={doc.Status}; confidence={doc.ConfidenceScore:F2}; engine={result.Note}");
        TelemetryService.IncrementProcessed();
        if (doc.Status == DocumentStatus.Validated)
            TelemetryService.IncrementValidated();

        var msg = doc.Status switch
        {
            DocumentStatus.Validated => $"✓ Validado ({doc.ConfidenceScore:P0})",
            DocumentStatus.ReadyForReview => $"⚠ Para revisao ({doc.ConfidenceScore:P0})",
            _ => doc.StatusDisplay
        };
        Application.Current?.Dispatcher.Invoke(() => ToastService.ShowInfo($"{doc.Filename}: {msg}"));
    }

    private static double InferConfidenceFromKeyCount(GlinerResult result)
    {
        if (result.Extraction is not JsonElement root) return 0.50;
        if (root.ValueKind != JsonValueKind.Object) return 0.50;

        int keys = root.EnumerateObject().Count();
        if (keys == 1)
        {
            var first = root.EnumerateObject().First().Value;
            if (first.ValueKind == JsonValueKind.Object) keys = first.EnumerateObject().Count();
            else if (first.ValueKind == JsonValueKind.Array) keys = first.GetArrayLength();
        }
        return Math.Min(0.94, 0.55 + (keys * 0.05));
    }

    private bool CrossValidate(Document doc, GlinerResult result, AppDbContext db)
    {
        try
        {
            var api = new BrasilApiService();

            if (doc.DocumentType == "Boleto" && !string.IsNullOrEmpty(result.OcrText))
            {
                var digitsOnly = Regex.Replace(result.OcrText, @"\D", "");
                var line = Regex.Match(digitsOnly, @"\d{47,48}");
                if (line.Success)
                {
                    var info = BrasilApiService.ParseBoleto(line.Value);
                    if (info != null && info.IsValid)
                    {
                        doc.ExtractedJson = JsonSerializer.Serialize(new
                        {
                            valor = info.Value,
                            vencimento_original = info.DueDate?.ToString("yyyy-MM-dd"),
                            linha_digitavel = line.Value,
                            banco = info.BankCode
                        });
                        return true;
                    }
                }
            }

            if (!string.IsNullOrEmpty(result.OcrText))
            {
                var cnpj = Regex.Match(result.OcrText, @"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}");
                if (cnpj.Success && Validators.ValidateCnpj(cnpj.Value))
                    return true;
            }
        }
        catch { }
        return false;
    }

    private void AutoCategorize(Document doc, GlinerResult result, AppDbContext db)
    {
        if (string.IsNullOrEmpty(result.OcrText)) return;
        try
        {
            var accounts = db.Accounts.Where(a => !string.IsNullOrEmpty(a.AutoClassificationRegex)).ToList();
            foreach (var acc in accounts)
            {
                if (Regex.IsMatch(result.OcrText, acc.AutoClassificationRegex!))
                {
                    doc.ChartOfAccountId = acc.Id;
                    return;
                }
            }
        }
        catch { }
    }

    private static void MarkAsError(int docId, string errorMessage)
    {
        try
        {
            using var db = new AppDbContext();
            var doc = db.Documents.Find(docId);
            if (doc != null)
            {
                doc.Status = DocumentStatus.Error;
                doc.OcrText = errorMessage;
                doc.ProcessedAt = DateTime.Now;
                db.SaveChanges();
            }
        }
        catch { }

        Application.Current?.Dispatcher.Invoke(() =>
            ToastService.ShowError($"Erro no processamento: {errorMessage}"));
    }

    public void Shutdown() => _cts.Cancel();

    private static string[] GetLabelsForType(string docType) => docType switch
    {
        "Boleto" => new[] { "valor", "vencimento", "beneficiario", "pagador", "codigo_barras" },
        "Holerite" => new[] { "empregador", "empregado", "competencia", "salario_liquido", "total_proventos" },
        "ExtratoBancario" => new[] { "banco", "agencia", "conta", "saldo_inicial", "saldo_final" },
        "DARF" => new[] { "codigo_receita", "valor_total", "vencimento", "cnpj_contribuinte" },
        "DANFE" => new[] { "chave_acesso", "numero_nota", "valor_total", "cnpj_emitente", "data_emissao" },
        _ => new[] { "cnpj_emitente", "nome_emitente", "valor_total", "data_emissao", "numero_nota" }
    };

    private record QueueItem(int DocId, string FilePath, string DocType);
}
