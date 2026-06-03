using System.IO;
using System.Text;

namespace OContabil.Services;

/// <summary>
/// Extração de texto leve para arquivos comuns (txt/xml/csv/json).
/// Para PDFs e imagens delegamos ao bridge Python (com OCR), pois requer
/// runtimes externos. Esta classe é usada apenas como insumo para o motor
/// ONNX nativo quando ele está ativo.
/// </summary>
public static class DocumentTextExtractor
{
    private static readonly string[] _textExtensions = { ".txt", ".xml", ".csv", ".json", ".ofx", ".rem", ".ret" };

    public static bool CanReadNatively(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return _textExtensions.Contains(ext);
    }

    public static async Task<string> ReadAsync(string path, CancellationToken ct = default)
    {
        if (!File.Exists(path)) return string.Empty;

        // PDF não é lido como texto bruto (evita lixo binário): retorna vazio para
        // o pipeline cair no OCR (Python/Tesseract). A extração da camada de texto
        // de PDFs digitais exige uma lib PDF auditada — não adicionada aqui para não
        // puxar pacote não-verificado (princípio anti-supply-chain).
        if (Path.GetExtension(path).ToLowerInvariant() == ".pdf")
            return string.Empty;

        // .NET 8 não traz a code page 1252 por padrão (GetEncoding(1252) lança sem
        // CodePagesEncodingProvider). UTF-8 + Latin1 cobrem documentos PT-BR; o 1252
        // entra só se disponível. Construção guardada — nunca quebra a leitura.
        Encoding[] encodings;
        try { encodings = new[] { Encoding.UTF8, Encoding.Latin1, Encoding.GetEncoding(1252) }; }
        catch { encodings = new[] { Encoding.UTF8, Encoding.Latin1 }; }
        foreach (var enc in encodings)
        {
            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                using var reader = new StreamReader(stream, enc, detectEncodingFromByteOrderMarks: true);
                return await reader.ReadToEndAsync(ct);
            }
            catch { }
        }
        return string.Empty;
    }
}
