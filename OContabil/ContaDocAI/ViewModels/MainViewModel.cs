using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace ContaDocAI.ViewModels;

public partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    private string currentPage = "Dashboard";

    [ObservableProperty]
    private string pageTitle = "Dashboard";

    [ObservableProperty]
    private string pageSubtitle = "Visao geral do processamento";

    [ObservableProperty]
    private int validationBadge = 5;

    [RelayCommand]
    private void Navigate(string page)
    {
        CurrentPage = page;
        (PageTitle, PageSubtitle) = page switch
        {
            "Dashboard" => ("Dashboard", "Visao geral do processamento"),
            "Upload" => ("Upload de Documentos", "Enviar documentos para processamento com GLiNER 2"),
            "Validation" => ("Estacao de Validacao", "Revise e valide os dados extraidos pela IA"),
            "Clients" => ("Gestao de Clientes", "Cadastro e configuracao das empresas atendidas"),
            "Settings" => ("Configuracoes", "Configuracoes do sistema e integracoes"),
            _ => ("Dashboard", "Visao geral do processamento"),
        };
    }
}
