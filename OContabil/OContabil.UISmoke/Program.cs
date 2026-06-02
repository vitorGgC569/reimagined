using System;
using System.IO;
using System.Linq;
using System.Threading;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.Core.Definitions;
using FlaUI.Core.Capturing;
using FlaUI.Core.Input;
using FlaUI.UIA3;

// Runtime validation of the wired WebView2 design: launch, log in via the web
// form with the seeded admin (drives the real C# AuthService bridge), then
// navigate every screen and screenshot it to prove real data hydrates.
// Inputs are React-controlled, so we focus + send real keystrokes (UIA
// SetValue would not trigger React's onChange).
class Program
{
    const string Exe = @"C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OContabil\OContabil\bin\Debug\net8.0-windows\OContabil.exe";
    static readonly string ShotDir = @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_shots";

    static int Main()
    {
        Directory.CreateDirectory(ShotDir);
        foreach (var f in Directory.GetFiles(ShotDir, "*.png")) { try { File.Delete(f); } catch { } }

        var auto = new UIA3Automation();
        var app = Application.Launch(Exe);
        Thread.Sleep(13000); // WebView2 init + babel transpile + first render

        Window win = null;
        try { win = app.GetMainWindow(auto, TimeSpan.FromSeconds(20)); } catch (Exception e) { Console.WriteLine("win: " + e.Message); }
        if (win != null) { try { win.Focus(); } catch { } }
        Shot("w01_login");

        if (win != null)
        {
            var edits = Retry(() => { var l = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Edit)).ToList(); return l.Count >= 2 ? l : null; }, 14);
            if (edits != null && edits.Count >= 2)
            {
                TypeInto(edits[0], "admin");
                TypeInto(edits[1], "admin");
                Console.WriteLine("creds digitadas (" + edits.Count + " campos Edit)");
            }
            else Console.WriteLine("inputs nao encontrados: " + (edits?.Count ?? -1));
            Shot("w02_creds");

            var btn = win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button))
                         .FirstOrDefault(el => (el.Name ?? "").Trim().Equals("Entrar", StringComparison.OrdinalIgnoreCase));
            if (btn != null) { try { btn.AsButton().Invoke(); } catch { try { btn.Click(); } catch { } } Console.WriteLine("Entrar clicado"); }
            else Console.WriteLine("botao Entrar nao encontrado");
            Thread.Sleep(7000); // login + hydrate (6 bridge calls)
        }
        Shot("w03_dashboard");

        if (win != null)
        {
            string[] alvos = { "Documentos", "Clientes", "Schemas", "Exportações", "Usuários", "Painel" };
            foreach (var alvo in alvos)
            {
                var item = win.FindAllDescendants().FirstOrDefault(el => (el.Name ?? "").Trim().Equals(alvo, StringComparison.OrdinalIgnoreCase));
                if (item != null)
                {
                    try { item.Click(); Thread.Sleep(2400); Shot("w_" + Ascii(alvo)); Console.WriteLine("nav " + alvo); }
                    catch (Exception e) { Console.WriteLine("nav " + alvo + " falhou: " + e.Message); }
                }
                else Console.WriteLine("item nao encontrado: " + alvo);
            }
        }

        try { app.Close(); } catch { }
        try { if (!app.HasExited) app.Kill(); } catch { }
        auto.Dispose();
        return 0;
    }

    static void TypeInto(AutomationElement el, string text)
    {
        try
        {
            el.Click();            // click to focus the web input
            Thread.Sleep(200);
            Keyboard.Type(text);   // real keystrokes -> React onChange fires
            Thread.Sleep(200);
        }
        catch (Exception e) { Console.WriteLine("type: " + e.Message); }
    }

    static string Ascii(string s) => s.Replace("ç", "c").Replace("õ", "o").Replace("á", "a").Replace("ú", "u").Replace("ã", "a");

    static void Shot(string n)
    {
        try { Capture.Screen().ToFile(Path.Combine(ShotDir, n + ".png")); Console.WriteLine("shot " + n); }
        catch (Exception e) { Console.WriteLine("shot " + n + ": " + e.Message); }
    }

    static T Retry<T>(Func<T> f, int sec) where T : class
    {
        var end = DateTime.UtcNow.AddSeconds(sec);
        while (DateTime.UtcNow < end) { try { var r = f(); if (r != null) return r; } catch { } Thread.Sleep(300); }
        return null;
    }
}
