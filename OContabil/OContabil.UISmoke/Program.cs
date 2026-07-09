using System;
using System.IO;
using System.Linq;
using System.Threading;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.Core.Definitions;
using FlaUI.Core.Capturing;
using FlaUI.Core.Input;
using FlaUI.Core.WindowsAPI;
using FlaUI.UIA3;

// Runtime validation harness for the wired WebView2 UI.
//   (no args) -> login + navigate every screen, screenshot each (data hydration)
//   "io"      -> login + upload a real document (GLiNER pipeline) + export CSV
// React-controlled inputs need real keystrokes; native file dialogs are driven
// by typing the path into the focused filename field + Enter.
class Program
{
    static readonly string Exe = Environment.GetEnvironmentVariable("OCONTABIL_EXE")
        ?? @"C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\ocontabil-gaps\OContabil\OContabil\bin\Debug\net8.0-windows\OContabil.exe";
    static readonly string ShotDir = @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_shots";
    const StringComparison OIC = StringComparison.OrdinalIgnoreCase;

    static int Main(string[] args)
    {
        bool io = args.Any(a => a.Equals("io", OIC));
        bool org = args.Any(a => a.Equals("org", OIC));
        bool gate = args.Any(a => a.Equals("gate", OIC)); // licenca ausente/vencida -> login bloqueado
        bool cli = args.Any(a => a.Equals("cli", OIC));   // cadastro real de cliente
        Directory.CreateDirectory(ShotDir);
        var auto = new UIA3Automation();
        var app = Application.Launch(Exe);
        Thread.Sleep(13000);

        Window win = null;
        try { win = app.GetMainWindow(auto, TimeSpan.FromSeconds(20)); } catch (Exception e) { Console.WriteLine("win: " + e.Message); }
        if (win == null) { Console.WriteLine("sem janela"); auto.Dispose(); return 1; }
        try { win.Focus(); } catch { }

        Login(win);
        if (gate) { Thread.Sleep(1500); Shot("gate0_login"); }
        else if (cli) ClientsFlow(win);
        else if (org) OrganizeFlow(win);
        else if (io) UploadExportFlow(win);
        else NavFlow(win);

        try { app.Close(); } catch { }
        try { if (!app.HasExited) app.Kill(); } catch { }
        auto.Dispose();
        return 0;
    }

    static void Login(Window win)
    {
        // Seed novo (seguro): DB fresco gera senha temporaria aleatoria gravada em
        // FIRST_ACCESS.txt + MustChangePassword. Le a temporaria de la; "admin" e so
        // fallback para DBs antigos. Override manual: OCONTABIL_PASS.
        var pass = Environment.GetEnvironmentVariable("OCONTABIL_PASS") ?? ReadTempPassword() ?? "admin";
        Console.WriteLine("senha usada: " + (pass == "admin" ? "admin (fallback)" : "FIRST_ACCESS.txt"));

        var edits = Retry(() => { var l = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Edit)).ToList(); return l.Count >= 2 ? l : null; }, 14);
        if (edits == null) { Console.WriteLine("inputs login nao encontrados"); return; }
        TypeInto(edits[0], "admin");
        TypeInto(edits[1], pass);
        var btn = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(el => (el.Name ?? "").Trim().Equals("Entrar", OIC));
        if (btn != null) { try { btn.AsButton().Invoke(); } catch { try { btn.Click(); } catch { } } }
        Console.WriteLine("login enviado");
        Thread.Sleep(7000);

        // Gate de primeiro acesso ("Defina uma nova senha"): preenche nova+confirmacao
        // e salva. Valida em runtime o fluxo MustChangePassword -> password.change.
        var salvar = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button))
            .FirstOrDefault(b => (b.Name ?? "").Trim().StartsWith("Salvar e entrar", OIC));
        if (salvar != null)
        {
            const string nova = "Smoke#2026!Ui";
            var pwEdits = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Edit)).ToList();
            if (pwEdits.Count >= 2)
            {
                TypeInto(pwEdits[0], nova);
                TypeInto(pwEdits[1], nova);
                try { salvar.AsButton().Invoke(); } catch { try { salvar.Click(); } catch { } }
                Console.WriteLine("primeiro acesso: senha temporaria trocada");
                Thread.Sleep(6000);
            }
            else Console.WriteLine("gate de troca detectado mas inputs nao encontrados");
        }
    }

    static string ReadTempPassword()
    {
        try
        {
            var p = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "OContabil", "FIRST_ACCESS.txt");
            if (!File.Exists(p)) return null;
            var line = File.ReadAllLines(p).FirstOrDefault(l => l.StartsWith("Senha temporaria:", OIC));
            var pw = line?.Substring("Senha temporaria:".Length).Trim();
            return string.IsNullOrEmpty(pw) ? null : pw;
        }
        catch { return null; }
    }

    static void NavFlow(Window win)
    {
        Shot("w03_dashboard");
        foreach (var alvo in new[] { "Documentos", "Clientes", "Schemas", "Exportações", "Usuários", "Painel" })
        {
            var item = win.FindAllDescendants().FirstOrDefault(el => (el.Name ?? "").Trim().Equals(alvo, OIC));
            if (item != null) { try { item.Click(); Thread.Sleep(2400); Shot("w_" + Ascii(alvo)); Console.WriteLine("nav " + alvo); } catch (Exception e) { Console.WriteLine("nav " + alvo + ": " + e.Message); } }
            else Console.WriteLine("item nao encontrado: " + alvo);
        }
    }

    static void UploadExportFlow(Window win)
    {
        string testDoc = Environment.GetEnvironmentVariable("OCONTABIL_TESTDOC") ?? @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_test\boleto_atlantico.txt";
        string csvOut = @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_test\export_teste.csv";
        try { if (File.Exists(csvOut)) File.Delete(csvOut); } catch { }

        // ---------- UPLOAD -> GLiNER ----------
        Nav(win, "Documentos"); Thread.Sleep(1500);
        var importar = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(b => (b.Name ?? "").Trim().Equals("Importar", OIC));
        if (importar != null)
        {
            try { importar.AsButton().Invoke(); } catch { try { importar.Click(); } catch { } }
            Console.WriteLine("Importar clicado");
            Thread.Sleep(3000);                 // OpenFileDialog abre (campo Nome focado)
            Keyboard.Type(testDoc);
            Thread.Sleep(500);
            Keyboard.Type(VirtualKeyShort.RETURN);
            Console.WriteLine("arquivo informado: " + testDoc);
        }
        else Console.WriteLine("botao Importar NAO encontrado");
        Thread.Sleep(WaitMs(14000));            // enqueue + extracao (ONNX->Python->Regex) + refresh JS; override: OCONTABIL_SMOKE_WAITMS
        Shot("io1_documentos");

        // ---------- EXPORT -> CSV ----------
        Nav(win, "Exportações"); Thread.Sleep(1800);
        // Status -> "Todos os status" via <select> nativo (ComboBox) p/ incluir o doc
        try
        {
            var combos = win.FindAllDescendants(cf => cf.ByControlType(ControlType.ComboBox)).ToList();
            Console.WriteLine("combos encontrados: " + combos.Count);
            var statusCombo = combos.FirstOrDefault(c => { try { return (c.AsComboBox().SelectedItem?.Text ?? "").Contains("aprovados", OIC); } catch { return false; } });
            if (statusCombo != null) { statusCombo.AsComboBox().Select("Todos os status"); Thread.Sleep(900); Console.WriteLine("status -> Todos os status"); }
            else Console.WriteLine("combo de status nao achado");
        }
        catch (Exception e) { Console.WriteLine("status combo erro: " + e.Message); }
        var exportar = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(b => (b.Name ?? "").Trim().StartsWith("Exportar", OIC));
        if (exportar != null)
        {
            try { exportar.AsButton().Invoke(); } catch { try { exportar.Click(); } catch { } }
            Console.WriteLine("Exportar clicado");
            Thread.Sleep(3000);                 // SaveFileDialog
            Keyboard.Type(csvOut);
            Thread.Sleep(500);
            Keyboard.Type(VirtualKeyShort.RETURN);
            Thread.Sleep(1300);
            Keyboard.Type(VirtualKeyShort.RETURN); // eventual "substituir?"
            Console.WriteLine("csv informado: " + csvOut);
        }
        else Console.WriteLine("botao Exportar NAO encontrado");
        Thread.Sleep(3500);
        Shot("io2_export");
        Console.WriteLine("CSV existe? " + File.Exists(csvOut) + (File.Exists(csvOut) ? " (" + new FileInfo(csvOut).Length + " bytes)" : ""));
    }

    static void ClientsFlow(Window win)
    {
        Nav(win, "Clientes"); Thread.Sleep(2200);
        Shot("cli0_lista");

        var novo = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button))
            .FirstOrDefault(b => (b.Name ?? "").Trim().Equals("Novo cliente", OIC));
        if (novo == null) { Console.WriteLine("botao Novo cliente NAO encontrado"); return; }
        try { novo.AsButton().Invoke(); } catch { try { novo.Click(); } catch { } }
        Thread.Sleep(1800);

        // Edits no modal (a busca da lista fica atras): [0]=busca, [1]=razao, [2]=fantasia, [3]=cnpj
        var edits = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Edit)).ToList();
        Console.WriteLine("edits no modal: " + edits.Count);
        if (edits.Count >= 4)
        {
            TypeInto(edits[1], "Escritorio Teste Piloto Ltda");
            TypeInto(edits[3], "11.222.333/0001-81");
        }
        var cad = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button))
            .FirstOrDefault(b => (b.Name ?? "").Trim().Equals("Cadastrar", OIC));
        if (cad != null) { try { cad.AsButton().Invoke(); } catch { try { cad.Click(); } catch { } } Console.WriteLine("Cadastrar clicado"); }
        else Console.WriteLine("botao Cadastrar NAO encontrado");
        Thread.Sleep(4000);        // clients.create + __refreshData
        Shot("cli1_pos_cadastro");
    }

    static void Nav(Window win, string alvo)
    {
        var item = win.FindAllDescendants().FirstOrDefault(el => (el.Name ?? "").Trim().Equals(alvo, OIC));
        if (item != null) { try { item.Click(); } catch { } } else Console.WriteLine("nav alvo nao achado: " + alvo);
    }

    static void OrganizeFlow(Window win)
    {
        string testDoc = Environment.GetEnvironmentVariable("OCONTABIL_TESTDOC") ?? @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_test\boleto_atlantico.txt";
        string root = @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_org";
        try { if (Directory.Exists(root)) Directory.Delete(root, true); } catch { }

        Nav(win, "Documentos"); Thread.Sleep(1500);
        var importar = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(b => (b.Name ?? "").Trim().Equals("Importar", OIC));
        if (importar != null)
        {
            try { importar.AsButton().Invoke(); } catch { }
            Thread.Sleep(3000); Keyboard.Type(testDoc); Thread.Sleep(500); Keyboard.Type(VirtualKeyShort.RETURN);
            Console.WriteLine("upload p/ ter o que organizar");
        }
        Thread.Sleep(14000);
        Shot("org1_documentos");

        var organizar = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(b => (b.Name ?? "").Trim().Equals("Organizar", OIC));
        if (organizar != null) { try { organizar.AsButton().Invoke(); } catch { } Console.WriteLine("Organizar clicado"); }
        else Console.WriteLine("botao Organizar NAO encontrado");
        Thread.Sleep(4500); // organize.plan
        Shot("org2_preview");

        var go = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(b => (b.Name ?? "").Trim().StartsWith("Escolher pasta", OIC));
        if (go != null)
        {
            try { go.AsButton().Invoke(); } catch { }
            Console.WriteLine("execucao clicada (abre seletor de pasta)");
            Thread.Sleep(3500);
            Keyboard.Type(root); Thread.Sleep(600); Keyboard.Type(VirtualKeyShort.RETURN);
            Thread.Sleep(1500); Keyboard.Type(VirtualKeyShort.RETURN);
            Thread.Sleep(3500);
        }
        else Console.WriteLine("botao executar NAO encontrado");
        Shot("org3_report");
        Console.WriteLine("pasta organizada existe? " + Directory.Exists(root));
    }

    static void TypeInto(AutomationElement el, string text)
    { try { el.Click(); Thread.Sleep(200); Keyboard.Type(text); Thread.Sleep(200); } catch (Exception e) { Console.WriteLine("type: " + e.Message); } }

    static string Ascii(string s) => s.Replace("ç", "c").Replace("õ", "o").Replace("á", "a").Replace("ú", "u").Replace("ã", "a");
    static void Shot(string n) { try { Capture.Screen().ToFile(Path.Combine(ShotDir, n + ".png")); Console.WriteLine("shot " + n); } catch (Exception e) { Console.WriteLine("shot " + n + ": " + e.Message); } }
    static int WaitMs(int def) => int.TryParse(Environment.GetEnvironmentVariable("OCONTABIL_SMOKE_WAITMS"), out var v) && v > 0 ? v : def;

    static T Retry<T>(Func<T> f, int sec) where T : class { var end = DateTime.UtcNow.AddSeconds(sec); while (DateTime.UtcNow < end) { try { var r = f(); if (r != null) return r; } catch { } Thread.Sleep(300); } return null; }
}
