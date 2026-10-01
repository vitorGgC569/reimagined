using ContaDocAI.Models;

namespace ContaDocAI.Services;

public static class MockDataService
{
    public static int ProcessedToday => 342;
    public static string ProcessedTodayTrend => "+18%";
    public static int AwaitingValidation => 5;
    public static int ReadingErrors => 2;
    public static double AccuracyRate => 96.4;
    public static int TotalClients => 47;
    public static string AvgProcessingTime => "1.8s";

    public static List<(string Day, int Count)> VolumeChart => new()
    {
        ("23/02", 287), ("24/02", 312), ("25/02", 198), ("26/02", 456),
        ("27/02", 389), ("28/02", 401), ("01/03", 523), ("02/03", 478),
        ("03/03", 267), ("04/03", 356), ("05/03", 512), ("06/03", 445),
        ("07/03", 389), ("08/03", 342),
    };

    public static List<Client> Clients => new()
    {
        new() { Id = 1, Name = "Construtora Horizonte Ltda", Cnpj = "12.345.678/0001-90", Color = "#6366f1", DocsMonth = 450, DocsValidated = 432, PendingDocs = 18 },
        new() { Id = 2, Name = "Restaurante Sabor e Arte ME", Cnpj = "23.456.789/0001-01", Color = "#8b5cf6", DocsMonth = 380, DocsValidated = 380, PendingDocs = 0 },
        new() { Id = 3, Name = "Tech Solutions Informatica Ltda", Cnpj = "34.567.890/0001-12", Color = "#06b6d4", DocsMonth = 612, DocsValidated = 590, PendingDocs = 22 },
        new() { Id = 4, Name = "Farmacia Popular Sao Joao", Cnpj = "45.678.901/0001-23", Color = "#10b981", DocsMonth = 890, DocsValidated = 878, PendingDocs = 12 },
        new() { Id = 5, Name = "Auto Pecas Nacional Eireli", Cnpj = "56.789.012/0001-34", Color = "#f59e0b", DocsMonth = 234, DocsValidated = 234, PendingDocs = 0 },
        new() { Id = 6, Name = "Clinica Odontologica Sorrisos", Cnpj = "67.890.123/0001-45", Color = "#ef4444", DocsMonth = 156, DocsValidated = 150, PendingDocs = 6 },
        new() { Id = 7, Name = "Supermercado Bom Preco Ltda", Cnpj = "78.901.234/0001-56", Color = "#3b82f6", DocsMonth = 1240, DocsValidated = 1200, PendingDocs = 40 },
        new() { Id = 8, Name = "Academia Corpo em Forma ME", Cnpj = "89.012.345/0001-67", Color = "#ec4899", DocsMonth = 98, DocsValidated = 98, PendingDocs = 0 },
    };

    public static List<(string Time, string Action, string Client, string Status, string Doc)> RecentActivity => new()
    {
        ("18:30", "NF processada", "Construtora Horizonte", "Success", "NF_2026_001234.pdf"),
        ("18:28", "Boleto extraido", "Restaurante Sabor e Arte", "Warning", "boleto_energia_mar2026.pdf"),
        ("18:15", "PIX validado", "Tech Solutions", "Success", "comprovante_pix_08mar.jpg"),
        ("17:50", "DARF processado", "Farmacia Popular", "Success", "DARF_IRPJ_fev2026.pdf"),
        ("17:30", "Erro OCR", "Clinica Sorrisos", "Error", "recibo_torto_scan.jpg"),
        ("17:15", "Extrato extraido", "Supermercado Bom Preco", "Success", "extrato_banco_fev2026.pdf"),
        ("16:45", "Lote processado", "Auto Pecas Nacional", "Success", "12 documentos"),
        ("16:20", "NF validada", "Academia Corpo em Forma", "Success", "NF_servico_0089.pdf"),
    };

    public static List<QueueItem> ProcessingQueue => new()
    {
        new() { Id = "Q-001", Filename = "lote_nfs_marco.zip", Size = "12.4 MB", FilesCount = 25, Status = "processing", Progress = 68 },
        new() { Id = "Q-002", Filename = "NF_servico_0145.pdf", Size = "340 KB", FilesCount = 1, Status = "processing", Progress = 90 },
        new() { Id = "Q-003", Filename = "fotos_whatsapp_recibos.zip", Size = "8.2 MB", FilesCount = 15, Status = "queued", Progress = 0 },
        new() { Id = "Q-004", Filename = "extrato_itau_jan.pdf", Size = "1.1 MB", FilesCount = 1, Status = "completed", Progress = 100 },
        new() { Id = "Q-005", Filename = "guias_impostos_fev.pdf", Size = "2.3 MB", FilesCount = 8, Status = "completed", Progress = 100 },
    };

    public static List<Document> ValidationQueue => new()
    {
        new()
        {
            Id = "DOC-001", Filename = "NF_2026_001234.pdf", ClientName = "Construtora Horizonte Ltda", ClientId = 1,
            DocumentType = "Nota Fiscal", UploadedAt = new DateTime(2026, 3, 8, 18, 30, 0), Status = "ready_for_review",
            OcrText = @"NOTA FISCAL DE SERVICOS ELETRONICA - NFS-e

Razao Social: Construtora Horizonte Ltda
CNPJ: 12.345.678/0001-90
Inscricao Municipal: 1234567

Data de Emissao: 08/03/2026
Numero da Nota: 001234

DISCRIMINACAO DOS SERVICOS:

Servicos de construcao civil - Etapa 3
Conforme contrato no 456/2025

VALOR TOTAL: R$ 15.750,00

ISS Retido: R$ 787,50 (5%)
Valor Liquido: R$ 14.962,50",
            ExtractedFields = new()
            {
                ["CNPJ Emissor"] = new() { Value = "12.345.678/0001-90", Confidence = 0.97, SpanStart = 45, SpanEnd = 63 },
                ["Razao Social"] = new() { Value = "Construtora Horizonte Ltda", Confidence = 0.95, SpanStart = 12, SpanEnd = 38 },
                ["Valor Total"] = new() { Value = "R$ 15.750,00", Confidence = 0.92, SpanStart = 120, SpanEnd = 132 },
                ["Data Emissao"] = new() { Value = "08/03/2026", Confidence = 0.98, SpanStart = 78, SpanEnd = 88 },
                ["Numero NF"] = new() { Value = "001234", Confidence = 0.99, SpanStart = 95, SpanEnd = 101 },
                ["Descricao"] = new() { Value = "Servicos de construcao civil - Etapa 3", Confidence = 0.85, SpanStart = 140, SpanEnd = 178 },
            }
        },
        new()
        {
            Id = "DOC-002", Filename = "boleto_energia_mar2026.pdf", ClientName = "Restaurante Sabor e Arte ME", ClientId = 2,
            DocumentType = "Boleto", UploadedAt = new DateTime(2026, 3, 8, 17, 45, 0), Status = "ready_for_review",
            OcrText = @"CONTA DE ENERGIA ELETRICA

CEMIG Distribuicao S.A.
CNPJ: 33.000.167/0001-01

Mes Referencia: MARCO/2026
Data de Vencimento: 15/03/2026

Consumo: 1.890 kWh

VALOR TOTAL: R$ 2.340,67

Codigo de Barras:
23793.38128 60000.000003 00000.000408",
            ExtractedFields = new()
            {
                ["CNPJ Emissor"] = new() { Value = "33.000.167/0001-01", Confidence = 0.94, SpanStart = 30, SpanEnd = 48 },
                ["Razao Social"] = new() { Value = "CEMIG Distribuicao S.A.", Confidence = 0.91, SpanStart = 10, SpanEnd = 32 },
                ["Valor Total"] = new() { Value = "R$ 2.340,67", Confidence = 0.88, SpanStart = 95, SpanEnd = 106 },
                ["Data Vencimento"] = new() { Value = "15/03/2026", Confidence = 0.96, SpanStart = 65, SpanEnd = 75 },
                ["Codigo Barras"] = new() { Value = "23793.38128 60000.000003", Confidence = 0.72, SpanStart = 180, SpanEnd = 210 },
            }
        },
        new()
        {
            Id = "DOC-003", Filename = "comprovante_pix_08mar.jpg", ClientName = "Tech Solutions Informatica Ltda", ClientId = 3,
            DocumentType = "Comprovante PIX", UploadedAt = new DateTime(2026, 3, 8, 16, 20, 0), Status = "ready_for_review",
            OcrText = @"COMPROVANTE DE TRANSFERENCIA PIX

Pagador: Tech Solutions Informatica Ltda
CNPJ: 34.567.890/0001-12
Banco: 001 - Banco do Brasil

Data/Hora: 08/03/2026 14:35:22

Valor da Transferencia: R$ 4.500,00

Chave: financeiro@fornecedor.com.br

Recebedor: Distribuidora Digital Express
CNPJ: 11.222.333/0001-44",
            ExtractedFields = new()
            {
                ["CNPJ Pagador"] = new() { Value = "34.567.890/0001-12", Confidence = 0.89, SpanStart = 20, SpanEnd = 38 },
                ["Nome Pagador"] = new() { Value = "Tech Solutions Informatica Ltda", Confidence = 0.93, SpanStart = 5, SpanEnd = 36 },
                ["Valor"] = new() { Value = "R$ 4.500,00", Confidence = 0.96, SpanStart = 80, SpanEnd = 91 },
                ["Data Pagamento"] = new() { Value = "08/03/2026", Confidence = 0.98, SpanStart = 55, SpanEnd = 65 },
                ["Chave PIX"] = new() { Value = "financeiro@fornecedor.com.br", Confidence = 0.78, SpanStart = 100, SpanEnd = 128 },
            }
        },
    };
}
