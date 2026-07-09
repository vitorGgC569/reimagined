using FluentAssertions;
using OContabil.Services;
using Xunit;

namespace OContabil.Tests;

public class PasswordHasherTests
{
    [Fact]
    public void Hash_ShouldProduceVerifiableHash()
    {
        var hash = PasswordHasher.Hash("MinhaSenha123!");
        hash.Should().StartWith("pbkdf2-sha256$");
        PasswordHasher.Verify("MinhaSenha123!", hash).Should().BeTrue();
    }

    [Fact]
    public void Hash_DifferentSaltProducesDifferentHashes()
    {
        var a = PasswordHasher.Hash("senha");
        var b = PasswordHasher.Hash("senha");
        a.Should().NotBe(b, "porque o salt é aleatório por hash");
        PasswordHasher.Verify("senha", a).Should().BeTrue();
        PasswordHasher.Verify("senha", b).Should().BeTrue();
    }

    [Fact]
    public void Verify_WrongPassword_ReturnsFalse()
    {
        var hash = PasswordHasher.Hash("certo");
        PasswordHasher.Verify("errado", hash).Should().BeFalse();
    }

    [Fact]
    public void Verify_LegacySha256_ReturnsTrueAndNeedsUpgrade()
    {
        // Formato legado real: SHA256(senha) em HEX minúsculo de 64 chars (o que a
        // versão original gravava; VerifyLegacySha256 compara em hex).
        using var sha = System.Security.Cryptography.SHA256.Create();
        var bytes = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes("admin"));
        var legacy = Convert.ToHexString(bytes).ToLowerInvariant();

        PasswordHasher.Verify("admin", legacy).Should().BeTrue();
        PasswordHasher.NeedsUpgrade(legacy).Should().BeTrue();
    }

    [Fact]
    public void NeedsUpgrade_ReturnsFalseForPbkdf2Hash()
    {
        var hash = PasswordHasher.Hash("x");
        PasswordHasher.NeedsUpgrade(hash).Should().BeFalse();
    }

    [Fact]
    public void Verify_NullOrEmpty_ReturnsFalse()
    {
        PasswordHasher.Verify("senha", null).Should().BeFalse();
        PasswordHasher.Verify("senha", "").Should().BeFalse();
        PasswordHasher.Verify("", "qualquer-hash").Should().BeFalse();
    }
}
