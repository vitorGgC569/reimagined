using System.Collections.Concurrent;

namespace OContabil.Services;

/// <summary>
/// Limita tentativas de login por usuário para evitar ataques de força bruta.
/// Após 5 tentativas mal sucedidas em 10 minutos, o usuário fica bloqueado
/// por 15 minutos. Em login bem-sucedido o contador é zerado.
/// </summary>
public sealed class LoginThrottleService
{
    private static readonly Lazy<LoginThrottleService> _shared =
        new(() => new LoginThrottleService());

    public static LoginThrottleService Instance => _shared.Value;

    private readonly ConcurrentDictionary<string, AttemptInfo> _attempts = new();
    private readonly object _gate = new();

    public int MaxAttempts { get; set; } = 5;
    public TimeSpan AttemptWindow { get; set; } = TimeSpan.FromMinutes(10);
    public TimeSpan LockoutDuration { get; set; } = TimeSpan.FromMinutes(15);

    public ThrottleResult CheckAllowed(string username)
    {
        username = NormalizeKey(username);
        if (string.IsNullOrEmpty(username))
            return ThrottleResult.Allowed();

        if (!_attempts.TryGetValue(username, out var info))
            return ThrottleResult.Allowed();

        if (info.LockedUntil.HasValue && info.LockedUntil > DateTime.UtcNow)
        {
            var remaining = info.LockedUntil.Value - DateTime.UtcNow;
            return ThrottleResult.Blocked(remaining);
        }

        return ThrottleResult.Allowed(Math.Max(0, MaxAttempts - info.Count));
    }

    public ThrottleResult RegisterFailure(string username)
    {
        username = NormalizeKey(username);
        if (string.IsNullOrEmpty(username))
            return ThrottleResult.Allowed();

        lock (_gate)
        {
            var now = DateTime.UtcNow;
            var info = _attempts.GetOrAdd(username, _ => new AttemptInfo());

            if (info.WindowStart == default || now - info.WindowStart > AttemptWindow)
            {
                info.WindowStart = now;
                info.Count = 0;
                info.LockedUntil = null;
            }

            info.Count++;

            if (info.Count >= MaxAttempts)
            {
                info.LockedUntil = now.Add(LockoutDuration);
                return ThrottleResult.Blocked(LockoutDuration);
            }

            return ThrottleResult.Allowed(MaxAttempts - info.Count);
        }
    }

    public void RegisterSuccess(string username)
    {
        username = NormalizeKey(username);
        if (string.IsNullOrEmpty(username))
            return;

        _attempts.TryRemove(username, out _);
    }

    public void Reset() => _attempts.Clear();

    private static string NormalizeKey(string? username) =>
        (username ?? string.Empty).Trim().ToLowerInvariant();

    private sealed class AttemptInfo
    {
        public DateTime WindowStart { get; set; }
        public int Count { get; set; }
        public DateTime? LockedUntil { get; set; }
    }
}

public readonly record struct ThrottleResult(bool IsAllowed, TimeSpan? BlockedFor, int RemainingAttempts)
{
    public static ThrottleResult Allowed(int remaining = int.MaxValue) =>
        new(true, null, remaining);

    public static ThrottleResult Blocked(TimeSpan duration) =>
        new(false, duration, 0);
}
