namespace ContaDocAI.Models;

public class Client
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string Cnpj { get; set; } = "";
    public string Color { get; set; } = "#6366f1";
    public int DocsMonth { get; set; }
    public int DocsValidated { get; set; }
    public int PendingDocs { get; set; }

    public string Initials => string.Join("", Name.Split(' ').Where(w => w.Length > 0).Take(2).Select(w => w[0]));
    public int ValidationPercent => DocsMonth > 0 ? (int)((double)DocsValidated / DocsMonth * 100) : 100;
    public string StatusText => PendingDocs == 0 ? "Em dia" : $"{PendingDocs} pendentes";
    public bool IsUpToDate => PendingDocs == 0;
}
