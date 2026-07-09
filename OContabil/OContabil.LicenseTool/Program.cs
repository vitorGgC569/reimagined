using System.Security.Cryptography;
using System.Text.Json;

// Ferramenta do DONO do produto (nunca distribuir ao testador):
//   keygen                          — gera o par ECDSA P-256 (private.pem fica SÓ nesta máquina)
//   issue --nome "X" --dias N       — emite license.ocl assinada com expiração
//
// A chave PÚBLICA correspondente vai embarcada em OContabil/Services/LicenseService.cs.
// Se você rodar keygen de novo (rotação), precisa atualizar a constante lá e recompilar.

var licDir = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
    "OContabil", "licensing");
Directory.CreateDirectory(licDir);
var privPath = Path.Combine(licDir, "private.pem");
var pubPath = Path.Combine(licDir, "public.pem");

if (args.Length == 0) { Help(); return 1; }

switch (args[0].ToLowerInvariant())
{
    case "keygen":
    {
        if (File.Exists(privPath) && !args.Contains("--force"))
        {
            Console.WriteLine($"Ja existe chave privada em {privPath}.");
            Console.WriteLine("Use 'keygen --force' apenas se quiser ROTACIONAR (licencas antigas param de valer apos recompilar o app).");
            return 1;
        }
        using var ecdsa = ECDsa.Create(ECCurve.NamedCurves.nistP256);
        File.WriteAllText(privPath, ecdsa.ExportECPrivateKeyPem());
        File.WriteAllText(pubPath, ecdsa.ExportSubjectPublicKeyInfoPem());
        Console.WriteLine($"Chave privada : {privPath}  (GUARDE — quem tem ela emite licencas)");
        Console.WriteLine($"Chave publica : {pubPath}");
        Console.WriteLine();
        Console.WriteLine("=== COLE ESTE BLOCO em OContabil/Services/LicenseService.cs (PublicKeyPem) ===");
        Console.WriteLine(File.ReadAllText(pubPath));
        return 0;
    }

    case "issue":
    {
        if (!File.Exists(privPath))
        {
            Console.WriteLine($"Chave privada nao encontrada ({privPath}). Rode 'keygen' primeiro.");
            return 1;
        }
        var nome = Arg("--nome");
        var diasStr = Arg("--dias");
        if (string.IsNullOrWhiteSpace(nome) || !int.TryParse(diasStr, out var dias) || dias <= 0)
        {
            Console.WriteLine("Uso: issue --nome \"Fulano / Escritorio X\" --dias 30 [--out caminho\\license.ocl]");
            return 1;
        }

        var payload = JsonSerializer.SerializeToUtf8Bytes(new
        {
            id = Guid.NewGuid().ToString("N"),
            nome,
            emitidaUtc = DateTime.UtcNow,
            expiraUtc = DateTime.UtcNow.AddDays(dias),
        });

        using var ecdsa = ECDsa.Create();
        ecdsa.ImportFromPem(File.ReadAllText(privPath));
        var sig = ecdsa.SignData(payload, HashAlgorithmName.SHA256);

        var envelope = JsonSerializer.Serialize(new
        {
            payload = Convert.ToBase64String(payload),
            signature = Convert.ToBase64String(sig),
        }, new JsonSerializerOptions { WriteIndented = true });

        var slug = new string(nome.Where(char.IsLetterOrDigit).Take(24).ToArray());
        var outPath = Arg("--out") ?? Path.Combine(licDir, $"license_{slug}_{DateTime.Now:yyyyMMdd}.ocl");
        File.WriteAllText(outPath, envelope);

        Console.WriteLine($"Licenca emitida para \"{nome}\" — valida por {dias} dia(s) (ate {DateTime.Now.AddDays(dias):dd/MM/yyyy}).");
        Console.WriteLine($"Arquivo: {outPath}");
        Console.WriteLine();
        Console.WriteLine("Entregue ao testador com a instrucao: renomear para license.ocl e colocar em");
        Console.WriteLine(@"  %LocalAppData%\OContabil\license.ocl   (ou na mesma pasta do OContabil.exe)");
        return 0;
    }

    default:
        Help();
        return 1;
}

string? Arg(string name)
{
    var i = Array.IndexOf(args, name);
    return (i >= 0 && i + 1 < args.Length) ? args[i + 1] : null;
}

void Help()
{
    Console.WriteLine("OContabil.LicenseTool — emissao de licencas de avaliacao (offline, assinadas)");
    Console.WriteLine("  keygen [--force]                        gera par de chaves (uma vez)");
    Console.WriteLine("  issue --nome \"X\" --dias 30 [--out p]    emite licenca com expiracao");
}
