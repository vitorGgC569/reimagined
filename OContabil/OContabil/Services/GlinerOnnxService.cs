using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace OContabil.Services;

/// <summary>
/// Infraestrutura de inferência ONNX Runtime (em-processo) para um futuro motor
/// nativo de extração. Carrega a sessão ONNX e o tokenizer e monta os tensores de
/// entrada. O DECODER de saída é específico do modelo: GLiNER2 (DeBERTa-v3 +
/// SentencePiece + decode por schema/count) exige um decoder fiel que ainda não
/// está implementado em C#. Enquanto isso, este motor reporta indisponível (nunca
/// fabrica spans) e o pipeline usa os motores determinístico/Python.
///
/// Modelo esperado em <c>Models/model.onnx</c> com <c>vocab.txt</c> ao lado.
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
            using var run = _session.Run(inputs); // valida que o modelo realmente roda
            InspectOutput(run);

            // Sem um decoder FIEL ao modelo (GLiNER2 = span/count por schema) não há
            // como mapear a saída em campos sem fabricar. Regra do projeto: sem stubs/
            // sem simulação → reportamos indisponível e o pipeline segue para o motor
            // determinístico/Python.
            return new GlinerResult
            {
                Success = false,
                Model = "onnx-native",
                Error = "Motor ONNX: decoder específico do modelo não implementado para este formato.",
                ThresholdUsed = threshold,
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

    /// <summary>
    /// Apenas inspeciona/loga o formato da saída do modelo (diagnóstico). NÃO
    /// decodifica entidades: o decode fiel do GLiNER2 (spans/count por schema) é
    /// específico do modelo e ainda não está implementado em C#. Não fabricamos
    /// posições — ver <see cref="PredictAsync"/>.
    /// </summary>
    private void InspectOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        try
        {
            var first = outputs.FirstOrDefault();
            var tensor = first?.AsTensor<float>();
            var dims = tensor != null ? string.Join(",", tensor.Dimensions.ToArray()) : "n/d";
            Log($"Saída ONNX recebida (dims=[{dims}]). Decoder específico do modelo " +
                "não implementado — encaminhando para o motor determinístico/Python.");
        }
        catch (Exception ex) { Log($"Inspeção de saída ONNX falhou: {ex.Message}"); }
    }

    public void Dispose()
    {
        _session?.Dispose();
        _session = null;
        _loaded = false;
    }
}
