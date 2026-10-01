using System.ComponentModel.DataAnnotations;
using System.Text.Json;

namespace OContabil.Models;

/// <summary>
/// Schema GLiNER associado a um tipo de documento. Pode ser global (IsSystem=true)
/// ou customizado para um cliente específico.
/// </summary>
public class DocumentSchema
{
    public int Id { get; set; }

    [Required, MaxLength(120)]
    public string Name { get; set; } = "";

    [Required, MaxLength(40)]
    public string DocumentType { get; set; } = "";

    public int? ClientId { get; set; }

    [Required]
    public string SchemaJson { get; set; } = "{}";

    public bool IsSystem { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public IEnumerable<string> FieldNames()
    {
        try
        {
            using var doc = JsonDocument.Parse(SchemaJson);
            if (doc.RootElement.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in doc.RootElement.EnumerateObject())
                {
                    if (prop.Value.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var f in prop.Value.EnumerateArray())
                        {
                            var raw = f.GetString() ?? "";
                            var name = raw.Split("::").FirstOrDefault() ?? raw;
                            yield return name;
                        }
                    }
                }
            }
        }
        finally { }
    }
}

public static class DocumentSchemaCatalog
{
    public static IEnumerable<DocumentSchema> BuiltIn()
    {
        yield return new DocumentSchema
        {
            Name = "Nota Fiscal Eletrônica (NF-e)",
            DocumentType = "NF-e",
            IsSystem = true,
            SchemaJson = """
            {
              "nota_fiscal": [
                "cnpj_emitente::str::CNPJ da empresa emitente (formato XX.XXX.XXX/XXXX-XX)",
                "nome_emitente::str::Razao social completa do emitente",
                "cnpj_destinatario::str::CNPJ do destinatario",
                "numero_nota::str::Numero da nota fiscal (serie e numero)",
                "data_emissao::str::Data de emissao (DD/MM/AAAA)",
                "valor_total::str::Valor total em reais",
                "valor_icms::str::Valor do ICMS destacado",
                "valor_ipi::str::Valor do IPI",
                "chave_acesso::str::Chave de acesso NF-e (44 digitos)",
                "natureza_operacao::str::Natureza da operacao",
                "descricao_produtos::str::Descricao dos produtos ou servicos"
              ]
            }
            """
        };

        yield return new DocumentSchema
        {
            Name = "DANFE - Documento Auxiliar da NF-e",
            DocumentType = "DANFE",
            IsSystem = true,
            SchemaJson = """
            {
              "danfe": [
                "chave_acesso::str::Chave de acesso NF-e com 44 digitos",
                "numero_nota::str::Numero da nota fiscal",
                "serie::str::Serie da nota fiscal",
                "data_emissao::str::Data de emissao (DD/MM/AAAA)",
                "data_saida::str::Data de saida ou entrada (DD/MM/AAAA)",
                "cnpj_emitente::str::CNPJ do emitente",
                "razao_social_emitente::str::Razao social do emitente",
                "cnpj_destinatario::str::CNPJ do destinatario",
                "razao_social_destinatario::str::Razao social do destinatario",
                "valor_total_produtos::str::Valor total dos produtos",
                "valor_total_nota::str::Valor total da nota",
                "valor_frete::str::Valor do frete",
                "natureza_operacao::str::Natureza da operacao",
                "protocolo_autorizacao::str::Numero do protocolo de autorizacao"
              ]
            }
            """
        };

        yield return new DocumentSchema
        {
            Name = "Boleto Bancário",
            DocumentType = "Boleto",
            IsSystem = true,
            SchemaJson = """
            {
              "boleto": [
                "linha_digitavel::str::Linha digitavel com 47 digitos",
                "codigo_barras::str::Codigo de barras com 44 digitos",
                "valor::str::Valor do boleto em reais",
                "vencimento::str::Data de vencimento (DD/MM/AAAA)",
                "beneficiario::str::Nome ou razao social do beneficiario",
                "cnpj_beneficiario::str::CNPJ do beneficiario",
                "pagador::str::Nome ou razao social do pagador",
                "banco::str::Nome do banco emissor",
                "agencia::str::Numero da agencia",
                "conta::str::Numero da conta",
                "nosso_numero::str::Identificador interno do boleto"
              ]
            }
            """
        };

        yield return new DocumentSchema
        {
            Name = "Extrato Bancário",
            DocumentType = "ExtratoBancario",
            IsSystem = true,
            SchemaJson = """
            {
              "extrato": [
                "banco::str::Nome do banco",
                "agencia::str::Numero da agencia",
                "conta::str::Numero da conta",
                "titular::str::Titular da conta",
                "periodo_inicio::str::Data inicial do periodo",
                "periodo_fim::str::Data final do periodo",
                "saldo_inicial::str::Saldo no inicio do periodo",
                "saldo_final::str::Saldo no final do periodo"
              ],
              "lancamentos": [
                "data::str::Data do lancamento (DD/MM/AAAA)",
                "descricao::str::Descricao ou historico do lancamento",
                "valor::str::Valor (positivo para credito, negativo para debito)",
                "tipo::str::C para credito, D para debito"
              ]
            }
            """
        };

        yield return new DocumentSchema
        {
            Name = "Holerite / Folha de Pagamento",
            DocumentType = "Holerite",
            IsSystem = true,
            SchemaJson = """
            {
              "holerite": [
                "empregador::str::Razao social do empregador",
                "cnpj_empregador::str::CNPJ do empregador",
                "empregado::str::Nome do empregado",
                "cpf_empregado::str::CPF do empregado",
                "cargo::str::Cargo do empregado",
                "competencia::str::Competencia (MM/AAAA)",
                "salario_base::str::Salario base",
                "total_proventos::str::Total de proventos",
                "total_descontos::str::Total de descontos",
                "salario_liquido::str::Valor liquido recebido",
                "base_inss::str::Base de calculo do INSS",
                "base_fgts::str::Base de calculo do FGTS",
                "valor_fgts::str::Valor do FGTS depositado"
              ]
            }
            """
        };

        yield return new DocumentSchema
        {
            Name = "DARF / Guia de Recolhimento",
            DocumentType = "DARF",
            IsSystem = true,
            SchemaJson = """
            {
              "darf": [
                "codigo_receita::str::Codigo da receita",
                "periodo_apuracao::str::Periodo de apuracao",
                "numero_referencia::str::Numero de referencia",
                "valor_principal::str::Valor principal",
                "valor_multa::str::Valor da multa",
                "valor_juros::str::Valor dos juros",
                "valor_total::str::Valor total a recolher",
                "data_vencimento::str::Data de vencimento",
                "cnpj_contribuinte::str::CNPJ do contribuinte"
              ]
            }
            """
        };
    }
}
