using System.Windows;

namespace OContabil.Views;

public partial class ShortcutsHelpDialog : Window
{
    public sealed record Shortcut(string Keys, string Description);

    public ShortcutsHelpDialog()
    {
        InitializeComponent();
        lstShortcuts.ItemsSource = new[]
        {
            new Shortcut("Alt + 1", "Ir para o Painel"),
            new Shortcut("Alt + 2", "Ir para Documentos"),
            new Shortcut("Alt + 3", "Ir para Clientes"),
            new Shortcut("Ctrl + T", "Alternar tema claro/escuro"),
            new Shortcut("F1", "Mostrar esta tela de ajuda"),
            new Shortcut("F5", "Recarregar a tela atual"),
            new Shortcut("Enter", "Confirmar diálogo / executar busca"),
            new Shortcut("Esc", "Fechar diálogo"),
            new Shortcut("Ctrl + N", "Novo (cliente/usuário/schema)"),
            new Shortcut("Ctrl + S", "Salvar (em diálogos de edição)"),
        };
    }

    private void OnClose(object sender, RoutedEventArgs e)
    {
        Close();
    }
}
