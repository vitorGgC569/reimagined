using System;
using System.IO;
using System.Linq;
using System.Threading;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.Core.Definitions;
using FlaUI.Core.Capturing;
using FlaUI.UIA3;

// Validation for the WebView2-hosted design: launch, let the web UI render,
// click the web "Entrar" button (creds pre-filled) to drive the real C# auth
// bridge, and screenshot the resulting dashboard.
class Program
{
    const string Exe = @"C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OContabil\OContabil\bin\Debug\net8.0-windows\OContabil.exe";
    static readonly string ShotDir = @"C:\Users\Oxta\AppData\Local\Temp\ocontabil_shots";

    static int Main()
    {
        Directory.CreateDirectory(ShotDir);
        var auto = new UIA3Automation();
        var app = Application.Launch(Exe);
        Thread.Sleep(11000); // WebView2 init + babel transpile + render

        Window win = null;
        try { win = app.GetMainWindow(auto, TimeSpan.FromSeconds(20)); } catch (Exception e) { Console.WriteLine("win: " + e.Message); }
        Shot("w01_login");

        if (win != null)
        {
            var btn = Retry(() => win.FindAllDescendants(cf => cf.ByControlType(ControlType.Button))
                                      .FirstOrDefault(el => (el.Name ?? "").Trim().Equals("Entrar", StringComparison.OrdinalIgnoreCase)), 10);
            if (btn != null)
            {
                try { btn.AsButton().Invoke(); } catch { try { btn.Click(); } catch (Exception e) { Console.WriteLine("click: " + e.Message); } }
                Console.WriteLine("Entrar clicado");
                Thread.Sleep(4000);
            }
            else Console.WriteLine("botao Entrar NAO encontrado");
        }
        Shot("w02_pos_login");

        // tenta navegar pelo menu lateral do design (itens web)
        if (win != null)
        {
            string[] alvos = { "Documentos", "Clientes", "Configurações" };
            foreach (var alvo in alvos)
            {
                var item = win.FindAllDescendants().FirstOrDefault(el => (el.Name ?? "").Trim().Equals(alvo, StringComparison.OrdinalIgnoreCase));
                if (item != null)
                {
                    try { item.Click(); Thread.Sleep(1500); Shot("w03_" + alvo); } catch { }
                }
            }
        }

        try { app.Close(); } catch { }
        try { if (!app.HasExited) app.Kill(); } catch { }
        auto.Dispose();
        return 0;
    }

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
