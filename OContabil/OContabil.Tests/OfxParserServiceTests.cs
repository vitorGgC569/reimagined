using System.IO;
using FluentAssertions;
using OContabil.Services;
using Xunit;

namespace OContabil.Tests;

public class OfxParserServiceTests
{
    private const string SampleOfx = @"OFXHEADER:100
DATA:OFXSGML
VERSION:102
SECURITY:NONE
ENCODING:USASCII
CHARSET:1252
COMPRESSION:NONE

<OFX>
<SIGNONMSGSRSV1>
<SONRS>
<STATUS><CODE>0</CODE><SEVERITY>INFO</SEVERITY></STATUS>
<DTSERVER>20240315120000</DTSERVER>
</SONRS>
</SIGNONMSGSRSV1>
<BANKMSGSRSV1>
<STMTTRNRS>
<STMTRS>
<CURDEF>BRL</CURDEF>
<BANKACCTFROM>
<BANKID>001</BANKID>
<ACCTID>12345-6</ACCTID>
<ACCTTYPE>CHECKING</ACCTTYPE>
</BANKACCTFROM>
<BANKTRANLIST>
<DTSTART>20240301</DTSTART>
<DTEND>20240331</DTEND>
<STMTTRN>
<TRNTYPE>CREDIT</TRNTYPE>
<DTPOSTED>20240305</DTPOSTED>
<TRNAMT>1500.00</TRNAMT>
<FITID>TX001</FITID>
<MEMO>Deposito</MEMO>
</STMTTRN>
<STMTTRN>
<TRNTYPE>DEBIT</TRNTYPE>
<DTPOSTED>20240310</DTPOSTED>
<TRNAMT>-250.50</TRNAMT>
<FITID>TX002</FITID>
<MEMO>Saque ATM</MEMO>
</STMTTRN>
</BANKTRANLIST>
<LEDGERBAL>
<BALAMT>1249.50</BALAMT>
<DTASOF>20240331</DTASOF>
</LEDGERBAL>
</STMTRS>
</STMTTRNRS>
</BANKMSGSRSV1>
</OFX>";

    [Fact]
    public void Parse_ValidOfx_ExtractsAllFields()
    {
        var tmp = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tmp, SampleOfx);
            var parser = new OfxParserService();
            var stmt = parser.Parse(tmp);

            stmt.BankId.Should().Be("001");
            stmt.AccountId.Should().Be("12345-6");
            stmt.Currency.Should().Be("BRL");
            stmt.LedgerBalance.Should().Be(1249.50m);
            stmt.Transactions.Should().HaveCount(2);
            stmt.Transactions[0].Amount.Should().Be(1500.00m);
            stmt.Transactions[0].Type.Should().Be("CREDIT");
            stmt.Transactions[1].Amount.Should().Be(-250.50m);
            stmt.Transactions[1].Memo.Should().Be("Saque ATM");
        }
        finally { File.Delete(tmp); }
    }

    [Fact]
    public void Parse_NonexistentFile_Throws()
    {
        var parser = new OfxParserService();
        Action act = () => parser.Parse("/tmp/nao-existe.ofx");
        act.Should().Throw<FileNotFoundException>();
    }
}
