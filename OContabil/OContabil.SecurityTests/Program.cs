using OContabil.Services;

// Suite de testes de seguranca (pentest deterministico). Cada Check mapeia um
// vetor a uma severidade. Criterio de parada do projeto: 0 alertas Medio/Alto/
// Critico => exit code 0.
int fail = 0, pass = 0, medHighCrit = 0;

void Check(string sev, string vector, bool ok, string detail = "")
{
    if (ok) { pass++; Console.WriteLine($"  [PASS] ({sev,4}) {vector}"); }
    else
    {
        fail++;
        if (sev is "MED" or "HIGH" or "CRIT") medHighCrit++;
        Console.WriteLine($"  [FAIL] ({sev,4}) {vector}  {detail}");
    }
}

Console.WriteLine("=== OContabil — Suite de Testes de Seguranca ===");
var root = Path.Combine(Path.GetTempPath(), "ocsec_root");

Console.WriteLine("\n[1] Path Traversal / confinamento de pastas (CWE-22)");
var t1 = SecurePath.CombineInside(root, "..", "..", "x");
Check("HIGH", "../.. neutralizado (resultado dentro da raiz)", t1 != null && SecurePath.IsInside(root, t1));
var seg = SecurePath.SanitizeSegment("../../etc/passwd");
Check("HIGH", "segmento sem '..' e sem separadores", !seg.Contains("..") && !seg.Contains('/') && !seg.Contains('\\'));
var t2 = SecurePath.CombineInside(root, @"C:\Windows\System32", "f.txt");
Check("HIGH", "caminho absoluto embutido fica confinado", t2 != null && SecurePath.IsInside(root, t2));
Check("HIGH", "IsInside rejeita caminho externo", !SecurePath.IsInside(root, @"C:\Windows\System32"));
Check("MED", "null byte removido do segmento", !SecurePath.SanitizeSegment("a\0b").Contains('\0'));
Check("MED", "nome reservado do Windows (CON) tratado", !string.Equals(SecurePath.SanitizeSegment("CON"), "CON", StringComparison.OrdinalIgnoreCase));

Console.WriteLine("\n[2] CSV / Formula Injection (CWE-1236)");
foreach (var p in new[] { "=cmd|'/C calc'!A0", "+1+1", "-2+3", "@SUM(A1)", "\tTAB" })
    Check("MED", $"neutraliza payload de formula ({p[..Math.Min(5, p.Length)].Replace("\t", "\\t")})", SecureCsv.Cell(p).StartsWith("'"));
Check("LOW", "valor legitimo permanece intacto", SecureCsv.Cell("Marilia Comercio Ltda") == "Marilia Comercio Ltda");
Check("MED", "celula com separador e' quotada", SecureCsv.Cell("a;b") is "\"a;b\"");
Check("MED", "aspas internas sao escapadas", SecureCsv.Cell("a\"b").Contains("\"\""));

Console.WriteLine("\n[3] Vazamento de PII/segredos em log (CWE-532)");
var red = SafeLog.Redact("Falha CNPJ 12.345.678/0001-90, CPF 123.456.789-09, mail a.b@x.com em C:\\Users\\Oxta\\doc.pdf");
Check("MED", "CNPJ redigido", red.Contains("[CNPJ]") && !red.Contains("12.345.678/0001-90"));
Check("MED", "CPF redigido", red.Contains("[CPF]") && !red.Contains("123.456.789-09"));
Check("MED", "e-mail redigido", red.Contains("[email]") && !red.Contains("a.b@x.com"));
Check("MED", "caminho de usuario redigido", red.Contains("[user]") && !red.Contains("Oxta"));

Console.WriteLine("\n[4] Cifragem de dados em repouso — DPAPI (CWE-311, Pilar 1)");
var secret = "{\"cnpj\":\"12.345.678/0001-90\",\"valor\":\"R$ 1.500,00\"}";
var enc = Crypto.Protect(secret);
Check("HIGH", "texto cifrado difere do claro (sem PII visivel)", enc != secret && !(enc ?? "").Contains("12.345.678") && !(enc ?? "").Contains("1.500"));
Check("HIGH", "conteudo marcado como cifrado", Crypto.IsEncrypted(enc));
Check("HIGH", "roundtrip decifra exatamente", Crypto.Unprotect(enc) == secret);
Check("MED", "dado legado em texto passa direto (compat)", Crypto.Unprotect("json-legado-sem-marcador") == "json-legado-sem-marcador");
Check("MED", "null/vazio nao quebram a cifragem", Crypto.Protect(null) == null && Crypto.Protect("") == "");
Check("LOW", "cifrar duas vezes nao duplica (idempotente)", Crypto.Unprotect(Crypto.Protect(enc)) == secret);

Console.WriteLine($"\n=== {pass} PASS / {fail} FAIL  (Med/Alto/Critico nao mitigados: {medHighCrit}) ===");
Console.WriteLine(medHighCrit == 0
    ? "OK: nenhum alerta de severidade Media/Alta/Critica."
    : $"BLOQUEIO: {medHighCrit} vetor(es) de severidade >= Media nao mitigado(s).");
return medHighCrit;
