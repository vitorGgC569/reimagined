using System;
using System.IO;
using System.Text;
using SkiaSharp;
using Tesseract;

namespace OContabil.Services;

/// <summary>
/// OCR 100% local (Tesseract) para imagens e PDFs escaneados. PDFs são
/// rasterizados página a página (PDFtoImage/PDFium) e cada imagem passa pelo
/// Tesseract (por+eng). Sem rede — soberania de dados preservada. Guardado:
/// nunca lança; em falha retorna vazio e o pipeline segue para o próximo motor.
/// </summary>
public static class OcrService
{
    private static readonly string TessData = Path.Combine(AppContext.BaseDirectory, "tessdata");

    public static bool IsAvailable =>
        Directory.Exists(TessData) && File.Exists(Path.Combine(TessData, "por.traineddata"));

    public static string ExtractText(string path)
    {
        if (!IsAvailable || !File.Exists(path)) return string.Empty;
        try
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();
            using var engine = new TesseractEngine(TessData, "por+eng", EngineMode.Default);
            var sb = new StringBuilder();

            if (ext == ".pdf")
            {
                using var fs = File.OpenRead(path);
                int page = 0;
                foreach (var bmp in PDFtoImage.Conversion.ToImages(fs))
                {
                    using (bmp)
                    using (var data = bmp.Encode(SKEncodedImageFormat.Png, 90))
                    using (var pix = Pix.LoadFromMemory(data.ToArray()))
                    using (var pg = engine.Process(pix))
                        sb.AppendLine(pg.GetText());

                    if (++page >= 30 || sb.Length > 2_000_000) break; // limites anti-DoS
                }
            }
            else
            {
                using var pix = Pix.LoadFromFile(path);
                using var pg = engine.Process(pix);
                sb.AppendLine(pg.GetText());
            }

            return sb.ToString().Trim();
        }
        catch (Exception ex) { SafeLog.Error("ocr", ex); return string.Empty; }
    }
}
