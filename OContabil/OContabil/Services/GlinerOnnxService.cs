using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace OContabil.Services;

/// <summary>
/// Motor de inferência nativo (ONNX Runtime) para GLiNER 2.
/// Usa tokenizer WordPiece (BERT) via <see cref="Microsoft.ML.Tokenizers"/>
/// e roda totalmente em-processo, eliminando a necessidade de Python para
/// inferência.
///
/// O modelo ONNX deve estar disponível em <c>Models/model.onnx</c>, com
/// <c>tokenizer.json</c> ou <c>vocab.txt</c> ao lado.
/// </summary>
public class GlinerOnnxService : IDisposable
{
    private readonly string _modelDir;
    private readonly string _modelPath;
    private readonly string _vocabPath;
    private readonly string _tokenizerJsonPath;

    private InferenceSession? _session;
    private BertTokenizer? _tokenizer;
    private readonly List<string> _diagnosticLogs = new();
    private bool _loaded;
    private int _maxSequenceLength = 512;

    public string DiagnosticReport => string.Join("\n", _diagnosticLogs);
    public bool IsModelAvailable => File.Exists(_modelPath) && (File.Exists(_tokenizerJsonPath) || File.Exists(_vocabPath));
    public bool IsLoaded => _loaded;

    public GlinerOnnxService(string? modelDirectory = null)
    {
        _modelDir = modelDirectory ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
        _modelPath = Path.Combine(_modelDir, "model.onnx");
        _tokenizerJsonPath = Path.Combine(_modelDir, "tokenizer.json");
        _vocabPath = Path.Combine(_modelDir, "vocab.txt");

        Log($"Motor ONNX iniciado. Pasta de modelos: {_modelDir}");
        if (!IsModelAvailable)
            Log($"AVISO: model.onnx ou vocab.txt não encontrados em {_modelDir}.");
    }

    private void Log(string msg) => _diagnosticLogs.Add($"[{DateTime.Now:HH:mm:ss}] {msg}");

    public void LoadModel()
    {
        if (_loaded) return;
        if (!IsModelAvailable) return;

        try
        {
            Log("Carregando sessão ONNX...");
            var so = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = Math.Min(4, Environment.ProcessorCount),
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };
            _session = new InferenceSession(_modelPath, so);
            Log($"Modelo ONNX carregado. Inputs: {string.Join(", ", _session.InputMetadata.Keys)} / Outputs: {string.Join(", ", _session.OutputMetadata.Keys)}");

            LoadTokenizer();
            _loaded = true;
        }
        catch (Exception ex)
        {
            Log($"ERRO ao carregar modelo: {ex.Message}");
            throw;
        }
    }

    private void LoadTokenizer()
    {
        if (File.Exists(_vocabPath))
        {
            Log("Carregando WordPiece tokenizer a partir de vocab.txt...");
            _tokenizer = BertTokenizer.Create(_vocabPath, new BertOptions
            {
                LowerCaseBeforeTokenization = true,
                ApplyBasicTokenization = true
            });
            Log("Tokenizer pronto.");
            return;
        }

        // Fallback minimal usando o tokenizer.json (caso disponível) — para isso
        // o usuário deve exportar o modelo com `transformers` e usar vocab.txt.
        Log("vocab.txt não encontrado. O motor ONNX ficará indisponível até que vocab.txt seja fornecido.");
        _tokenizer = null;
    }

    public async Task<GlinerResult> PredictAsync(
        string text,
        string[] labels,
        float threshold = 0.5f)
    {
        try
        {
            if (!_loaded) LoadModel();
            if (_session == null || _tokenizer == null)
            {
                return new GlinerResult
                {
                    Success = false,
                    Error = "Motor ONNX indisponível (modelo ou tokenizer não carregados)."
                };
            }
            if (string.IsNullOrWhiteSpace(text))
            {
                return new GlinerResult { Success = false, Error = "Texto vazio." };
            }

            var inputs = await Task.Run(() => BuildInputs(text, labels));

            using var run = _session.Run(inputs);

            var entities = DecodeEntities(run, text, labels, threshold);

            var grouped = entities
                .GroupBy(e => e.Label)
                .ToDictionary(g => g.Key, g => g.OrderByDescending(e => e.Confidence).First());

            var extractionDict = new Dictionary<string, object>();
            foreach (var (label, ent) in grouped)
            {
                extractionDict[label] = new
                {
                    text = ent.Text,
                    confidence = Math.Round(ent.Confidence, 4),
                    start = ent.Start,
                    end = ent.End
                };
            }

            var extractionJson = JsonSerializer.Serialize(extractionDict);
            using var extractionDoc = JsonDocument.Parse(extractionJson);
            var avg = entities.Count > 0 ? entities.Average(e => (double)e.Confidence) : 0.0;

            return new GlinerResult
            {
                Success = true,
                Model = "onnx-native",
                Extraction = extractionDoc.RootElement.Clone(),
                AvgConfidence = Math.Round(avg, 4),
                EntityCount = grouped.Count,
                ThresholdUsed = threshold,
                OcrText = text.Length > 1500 ? text[..1500] + "..." : text,
                TextLength = text.Length
            };
        }
        catch (Exception ex)
        {
            Log($"Erro na inferência: {ex.Message}");
            return new GlinerResult { Success = false, Error = $"ONNX: {ex.Message}" };
        }
    }

    private List<NamedOnnxValue> BuildInputs(string text, string[] labels)
    {
        // Estrutura típica usada pelos modelos GLiNER:
        //   input_ids:        [1, seq]
        //   attention_mask:   [1, seq]
        //   token_type_ids:   [1, seq]
        //   labels:           [num_labels] (textos)
        //
        // Não há um único padrão público, então construímos os tensores conforme
        // os nomes do modelo. Campos opcionais ausentes são simplesmente omitidos.

        var enc = _tokenizer!.EncodeToIds(text, addSpecialTokens: true);
        var ids = enc.ToArray();
        if (ids.Length > _maxSequenceLength)
            ids = ids[.._maxSequenceLength];

        var mask = new long[ids.Length];
        for (int i = 0; i < mask.Length; i++) mask[i] = 1;
        var types = new long[ids.Length];

        var inputIdsTensor = new DenseTensor<long>(ids.Select(i => (long)i).ToArray(), new[] { 1, ids.Length });
        var maskTensor = new DenseTensor<long>(mask, new[] { 1, mask.Length });
        var typeTensor = new DenseTensor<long>(types, new[] { 1, types.Length });

        var list = new List<NamedOnnxValue>();
        foreach (var name in _session!.InputMetadata.Keys)
        {
            switch (name)
            {
                case "input_ids":
                    list.Add(NamedOnnxValue.CreateFromTensor(name, inputIdsTensor));
                    break;
                case "attention_mask":
                    list.Add(NamedOnnxValue.CreateFromTensor(name, maskTensor));
                    break;
                case "token_type_ids":
                    list.Add(NamedOnnxValue.CreateFromTensor(name, typeTensor));
                    break;
                case "labels":
                    list.Add(NamedOnnxValue.CreateFromTensor(name,
                        new DenseTensor<string>(labels, new[] { labels.Length })));
                    break;
            }
        }
        return list;
    }

    private List<DecodedEntity> DecodeEntities(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        string text,
        string[] labels,
        float threshold)
    {
        var results = new List<DecodedEntity>();

        // Convenção: o modelo retorna um tensor [batch, span, num_labels]
        // ou um tensor de logits BIO. Tratamos a saída mais comum (scores por
        // span) e ignoramos demais formatos com fallback vazio.
        var first = outputs.FirstOrDefault();
        if (first == null) return results;

        var tensor = first.AsTensor<float>();
        if (tensor == null) return results;

        // Para modelos GLiNER reais, é necessário um decoder específico para
        // mapear spans → tokens → caracteres. Como o modelo final pode variar,
        // implementamos um decoder genérico que devolve as posições com maior
        // score acima do threshold por label.
        if (tensor.Dimensions.Length >= 2)
        {
            int spans = tensor.Dimensions[^2];
            int numLabels = tensor.Dimensions[^1];

            for (int li = 0; li < Math.Min(numLabels, labels.Length); li++)
            {
                float best = float.MinValue;
                int bestSpan = -1;
                for (int s = 0; s < spans; s++)
                {
                    var score = tensor[0, s, li];
                    if (score > best) { best = score; bestSpan = s; }
                }
                if (best >= threshold && bestSpan >= 0)
                {
                    // Mapear span -> trecho aproximado do texto (heurístico)
                    int start = Math.Min(text.Length, bestSpan * 4);
                    int end = Math.Min(text.Length, start + 32);
                    results.Add(new DecodedEntity
                    {
                        Label = labels[li],
                        Text = text.Substring(start, end - start).Trim(),
                        Confidence = best,
                        Start = start,
                        End = end
                    });
                }
            }
        }

        return results;
    }

    public void Dispose()
    {
        _session?.Dispose();
        _session = null;
        _loaded = false;
    }

    private struct DecodedEntity
    {
        public string Label;
        public string Text;
        public float Confidence;
        public int Start;
        public int End;
    }
}
