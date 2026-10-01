using System;
using System.IO;
using System.Linq;

namespace OContabil.Services;

/// <summary>
/// Defesa contra Path Traversal (CWE-22). Sanitiza segmentos individuais
/// (remove separadores, "..", caracteres inválidos e nomes reservados do
/// Windows) e — o mais importante — garante que o caminho final resolvido
/// permaneça SEMPRE dentro da raiz autorizada. Qualquer tentativa de escapar
/// (../, links, caminhos absolutos embutidos) é rejeitada.
/// </summary>
public static class SecurePath
{
    private static readonly string[] Reserved =
    {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    };

    /// <summary>Limpa um único segmento de pasta/arquivo: sem separadores, sem "..".</summary>
    public static string SanitizeSegment(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return "_";
        s = s.Replace('/', '-').Replace('\\', '-');
        foreach (var ch in Path.GetInvalidFileNameChars()) s = s.Replace(ch, '-');
        // Neutraliza qualquer run de 2+ pontos ("..") em qualquer posição — preserva
        // o ponto único de extensão (ex.: nota.xml), mas elimina traversal residual.
        s = System.Text.RegularExpressions.Regex.Replace(s, @"\.{2,}", "_");
        s = s.Trim().Trim('.', ' ');
        if (s.Length == 0 || s == "." || s == "..") return "_";
        var stem = Path.GetFileNameWithoutExtension(s).ToUpperInvariant();
        if (Reserved.Contains(stem)) s = "_" + s;
        return s.Length > 120 ? s.Substring(0, 120) : s;
    }

    /// <summary>True se <paramref name="candidate"/> resolve para dentro de <paramref name="root"/>.</summary>
    public static bool IsInside(string root, string candidate)
    {
        try
        {
            var r = Path.GetFullPath(root)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            var c = Path.GetFullPath(candidate);
            return c.StartsWith(r, StringComparison.OrdinalIgnoreCase)
                || string.Equals(c.TrimEnd(Path.DirectorySeparatorChar),
                                 r.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }

    /// <summary>
    /// Combina segmentos (cada um sanitizado) sob a raiz e valida o resultado.
    /// Retorna null se, por qualquer razão, o caminho escapar da raiz — o chamador
    /// deve tratar null como "rejeitado" (nunca cair de volta para o caminho cru).
    /// </summary>
    public static string? CombineInside(string root, params string[] segments)
    {
        try
        {
            var path = Path.GetFullPath(root);
            foreach (var seg in segments)
                path = Path.Combine(path, SanitizeSegment(seg));
            return IsInside(root, path) ? path : null;
        }
        catch { return null; }
    }
}
