using System;
using System.IO;
using System.Text.RegularExpressions;

namespace OContabil.Services;

/// <summary>
/// Log de aplicação seguro. Grava em <c>%LocalAppData%\OContabil\logs</c> com
/// REDAÇÃO automática de dados sensíveis (CNPJ/CPF, e-mails, caminhos de usuário)
/// — nunca vaza PII fiscal nem segredos em arquivo de log. Nunca lança exceção:
/// registrar um erro jamais pode derrubar o fluxo principal.
/// </summary>
public static class SafeLog
{
    private static readonly string LogDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OContabil", "logs");
    private static readonly object Gate = new();

    private static readonly Regex Cnpj = new(@"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}", RegexOptions.Compiled);
    private static readonly Regex Cpf = new(@"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", RegexOptions.Compiled);
    private static readonly Regex Email = new(@"[\w.+-]+@[\w-]+\.[\w.-]+", RegexOptions.Compiled);
    private static readonly Regex UserPath = new(@"[A-Za-z]:\\Users\\[^\\/]+", RegexOptions.Compiled);

    public static void Error(string context, Exception ex) =>
        Write("ERRO", context, ex.GetType().Name + ": " + ex.Message);

    public static void Error(string context, string msg) => Write("ERRO", context, msg);
    public static void Warn(string context, string msg) => Write("AVISO", context, msg);
    public static void Info(string context, string msg) => Write("INFO", context, msg);

    private static void Write(string level, string context, string message)
    {
        try
        {
            Directory.CreateDirectory(LogDir);
            var line = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [{level}] {SanitizeSegmentLite(context)}: {Redact(message)}{Environment.NewLine}";
            lock (Gate)
                File.AppendAllText(Path.Combine(LogDir, $"app-{DateTime.Now:yyyyMMdd}.log"), line);
        }
        catch { /* logar nunca quebra o app */ }
    }

    /// <summary>Remove PII/segredos do texto antes de persistir.</summary>
    public static string Redact(string? s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        s = Cnpj.Replace(s, "[CNPJ]");
        s = Cpf.Replace(s, "[CPF]");
        s = Email.Replace(s, "[email]");
        s = UserPath.Replace(s, @"C:\Users\[user]");
        s = s.Replace("\r", " ").Replace("\n", " ");
        return s.Length > 2000 ? s.Substring(0, 2000) + "…" : s;
    }

    private static string SanitizeSegmentLite(string s) =>
        string.IsNullOrEmpty(s) ? "-" : s.Replace("\r", " ").Replace("\n", " ");
}
