using System.Text.Json;
using FluentAssertions;
using OContabil.Services;
using Xunit;

namespace OContabil.Tests;

public class RegexExtractionServiceTests
{
    [Fact]
    public void Extract_NFe_FindsKeyFields()
    {
        var text = @"NOTA FISCAL ELETRONICA
                     Numero: 123456
                     Serie: 1
                     Data Emissao: 15/03/2024
                     Valor Total: R$ 1.234,56
                     CNPJ Emitente: 11.222.333/0001-81";

        var result = RegexExtractionService.Extract(text, "NF-e", 0.5);
        result.Success.Should().BeTrue();
        result.Extraction.Should().NotBeNull();

        var root = (JsonElement)result.Extraction!;
        root.GetProperty("numero_nota").GetString().Should().Be("123456");
        root.GetProperty("data_emissao").GetString().Should().Contain("15/03/2024");
    }

    [Fact]
    public void Extract_Boleto_FindsLinhaDigitavel()
    {
        var text = "Linha digitavel: 34191.79001 01043.510047 91020.150008 5 84410026000";
        var result = RegexExtractionService.Extract(text, "Boleto", 0.5);
        result.Success.Should().BeTrue();
    }

    [Fact]
    public void Extract_EmptyText_ReturnsFailure()
    {
        var result = RegexExtractionService.Extract("", "NF-e", 0.5);
        result.Success.Should().BeFalse();
    }

    [Fact]
    public void Extract_UnknownDocType_StillWorks()
    {
        var text = "Algum texto qualquer";
        var result = RegexExtractionService.Extract(text, "Desconhecido", 0.5);
        // Não deve lançar, apenas retornar com poucos campos.
        result.Should().NotBeNull();
    }
}
