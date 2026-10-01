using FluentAssertions;
using OContabil.Services;
using Xunit;

namespace OContabil.Tests;

public class LoginThrottleServiceTests
{
    [Fact]
    public void NewUser_IsAllowedAndHasFullAttempts()
    {
        var svc = new LoginThrottleService();
        var r = svc.CheckAllowed("novo_usuario_" + Guid.NewGuid());
        r.IsAllowed.Should().BeTrue();
        r.RemainingAttempts.Should().BeGreaterThan(0);
    }

    [Fact]
    public void AfterFiveFailures_UserIsBlocked()
    {
        var svc = new LoginThrottleService();
        var user = "throttle_test_" + Guid.NewGuid();
        for (int i = 0; i < 5; i++) svc.RegisterFailure(user);

        var r = svc.CheckAllowed(user);
        r.IsAllowed.Should().BeFalse();
        r.BlockedFor.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public void RegisterSuccess_ResetsCounter()
    {
        var svc = new LoginThrottleService();
        var user = "reset_test_" + Guid.NewGuid();
        for (int i = 0; i < 3; i++) svc.RegisterFailure(user);
        svc.RegisterSuccess(user);

        var r = svc.CheckAllowed(user);
        r.IsAllowed.Should().BeTrue();
    }

    [Fact]
    public void RegisterFailure_CaseInsensitive()
    {
        var svc = new LoginThrottleService();
        var user = "case_test_" + Guid.NewGuid();
        for (int i = 0; i < 5; i++) svc.RegisterFailure(user.ToUpper());

        svc.CheckAllowed(user.ToLower()).IsAllowed.Should().BeFalse();
    }
}
