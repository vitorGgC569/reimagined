using FluentAssertions;
using OContabil.Services;
using Xunit;

namespace OContabil.Tests;

public class ValidatorsTests
{
    [Theory]
    [InlineData("11.222.333/0001-81", true)]
    [InlineData("11222333000181", true)]
    [InlineData("00.000.000/0001-91", true)]
    [InlineData("11.111.111/1111-11", false)] // all same digits
    [InlineData("12.345.678/0001-99", false)] // wrong check digit
    [InlineData("1234567890123", false)]      // too short
    [InlineData("", false)]
    [InlineData("abc", false)]
    public void ValidateCnpj_ReturnsExpected(string input, bool expected)
    {
        Validators.ValidateCnpj(input).Should().Be(expected);
    }
}
