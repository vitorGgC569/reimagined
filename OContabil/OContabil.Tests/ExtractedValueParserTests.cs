using FluentAssertions;
using OContabil.Services.Exports;
using Xunit;

namespace OContabil.Tests;

public class ExtractedValueParserTests
{
    [Fact]
    public void ParseMoney_BrazilianFormat_Returns_Decimal()
    {
        var json = "{ \"valor_total\": \"1.234,56\" }";
        ExtractedValueParser.ParseMoney(json, "valor_total").Should().Be(1234.56m);
    }

    [Fact]
    public void ParseMoney_USFormat_Returns_Decimal()
    {
        var json = "{ \"valor_total\": \"1234.56\" }";
        ExtractedValueParser.ParseMoney(json, "valor_total").Should().Be(1234.56m);
    }

    [Fact]
    public void ParseMoney_NumberLiteral_Returns_Decimal()
    {
        var json = "{ \"valor_total\": 99.95 }";
        ExtractedValueParser.ParseMoney(json, "valor_total").Should().Be(99.95m);
    }

    [Fact]
    public void ParseMoney_MissingField_FallsBackThroughCandidates()
    {
        var json = "{ \"valor\": \"50,00\" }";
        ExtractedValueParser.ParseMoney(json, "valor_total", "valor").Should().Be(50.00m);
    }

    [Fact]
    public void ParseMoney_NoMatch_ReturnsZero()
    {
        var json = "{}";
        ExtractedValueParser.ParseMoney(json, "valor_total").Should().Be(0m);
    }

    [Fact]
    public void ParseDate_BrazilianFormat_Returns_DateTime()
    {
        var json = "{ \"data_emissao\": \"15/03/2024\" }";
        var d = ExtractedValueParser.ParseDate(json, "data_emissao");
        d.Should().NotBeNull();
        d!.Value.Day.Should().Be(15);
        d.Value.Month.Should().Be(3);
        d.Value.Year.Should().Be(2024);
    }

    [Fact]
    public void ParseDate_NoMatch_ReturnsNull()
    {
        ExtractedValueParser.ParseDate("{}", "data_emissao").Should().BeNull();
    }

    [Fact]
    public void ReadField_RecursiveSearch()
    {
        var json = "{ \"emitente\": { \"nome\": \"ACME\" } }";
        ExtractedValueParser.ReadField(json, "nome").Should().Be("ACME");
    }
}
