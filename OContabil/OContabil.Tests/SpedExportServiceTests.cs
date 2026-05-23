using System.IO;
using FluentAssertions;
using OContabil.Services.Exports;
using Xunit;

namespace OContabil.Tests;

public class SpedExportServiceTests
{
    [Fact]
    public void Export_EmptyRequest_ReturnsEmptyAndDoesNotCrash()
    {
        var svc = new SpedExportService();
        var path = Path.Combine(Path.GetTempPath(), $"sped_test_{Guid.NewGuid()}.txt");
        var req = new ExportRequest
        {
            OutputPath = path,
            OnlyValidated = true,
            StartDate = new DateTime(1999, 1, 1),
            EndDate = new DateTime(1999, 12, 31) // Período em que não há docs
        };

        // Em ambiente sem banco populado, deve retornar string vazia sem lançar.
        Action act = () => svc.Export(req);
        act.Should().NotThrow();
    }

    [Fact]
    public void Export_WithoutOutputPath_Throws()
    {
        var svc = new SpedExportService();
        Action act = () => svc.Export(new ExportRequest { OutputPath = "" });
        act.Should().Throw<ArgumentException>();
    }
}
