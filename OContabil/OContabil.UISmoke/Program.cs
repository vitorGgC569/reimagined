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
    const string Exe = @"C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OContabil\OContabil\bin\Debug\net8.0-windows\OContabil.exe";
    static readonly string ShotDir = @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_shots";
    const StringComparison OIC = StringComparison.OrdinalIgnoreCase;

    static int Main(string[] args)
    {
        bool io = args.Any(a => a.Equals("io", OIC));
        bool org = args.Any(a => a.Equals("org", OIC));
        Directory.CreateDirectory(ShotDir);
        var auto = new UIA3Automation();
        var app = Application.Launch(Exe);
        Thread.Sleep(13000);

        Window win = null;
        try { win = app.GetMainWindow(auto, TimeSpan.FromSeconds(20)); } catch (Exception e) { Console.WriteLine("win: " + e.Message); }
        if (win == null) { Console.WriteLine("sem janela"); auto.Dispose(); return 1; }
        try { win.Focus(); } catch { }

        Login(win);
        if (org) OrganizeFlow(win);
        else if (io) UploadExportFlow(win);
        else NavFlow(win);

        try { app.Close(); } catch { }
        try { if (!app.HasExited) app.Kill(); } catch { }
        auto.Dispose();
        return 0;
    }

    static void Login(Window win)
    {
        var edits = Retry(() => { var l = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Edit)).ToList(); return l.Count >= 2 ? l : null; }, 14);
        if (edits == null) { Console.WriteLine("inputs login nao encontrados"); return; }
        TypeInto(edits[0], "admin");
        TypeInto(edits[1], "admin");
        var btn = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button)).FirstOrDefault(el => (el.Name ?? "").Trim().Equals("Entrar", OIC));
        if (btn != null) { try { btn.AsButton().Invoke(); } catch { try { btn.Click(); } catch { } } }
        Console.WriteLine("login enviado");
        Thread.Sleep(7000);
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
        Thread.Sleep(14000);                    // enqueue + extracao (ONNX->Python->Regex) + refresh JS (0/2.5/6s)
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
    static T Retry<T>(Func<T> f, int sec) where T : class { var end = DateTime.UtcNow.AddSeconds(sec); while (DateTime.UtcNow < end) { try { var r = f(); if (r != null) return r; } catch { } Thread.Sleep(300); } return null; }
}
