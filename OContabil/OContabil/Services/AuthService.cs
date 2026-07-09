using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services;

public class AuthService
{
    private readonly LoginThrottleService _throttle = LoginThrottleService.Instance;

    public User? CurrentUser { get; private set; }
    public bool IsLoggedIn => CurrentUser != null;

    public SessionService Session { get; } = new();

    public AuthService() { }

    public LoginOutcome Login(string username, string password)
    {
        var check = _throttle.CheckAllowed(username);
        if (!check.IsAllowed)
        {
            return LoginOutcome.Throttled(check.BlockedFor ?? TimeSpan.Zero);
        }

        using var db = new AppDbContext();
        var user = db.Users.FirstOrDefault(u => u.Username == username && u.IsActive);
        if (user == null)
        {
            _throttle.RegisterFailure(username);
            AuditLogger.Write(db, null, "auth.login.failed", "Users", null, $"Username inexistente: {username}");
            return LoginOutcome.InvalidCredentials();
        }

        if (!PasswordHasher.Verify(password, user.PasswordHash))
        {
            var fail = _throttle.RegisterFailure(username);
            AuditLogger.Write(db, user.Id, "auth.login.failed", "Users", user.Id, "Senha incorreta");
            if (!fail.IsAllowed)
                return LoginOutcome.Throttled(fail.BlockedFor ?? TimeSpan.Zero);
            return LoginOutcome.InvalidCredentials(fail.RemainingAttempts);
        }

        if (PasswordHasher.NeedsUpgrade(user.PasswordHash))
        {
            user.PasswordHash = PasswordHasher.Hash(password);
            db.SaveChanges();
        }

        CurrentUser = user;
        user.LastLoginAt = DateTime.UtcNow;
        db.SaveChanges();
        _throttle.RegisterSuccess(username);
        AuditLogger.Write(db, user.Id, "auth.login.success", "Users", user.Id, "Login bem sucedido");

        return LoginOutcome.Success(user);
    }

    public void Logout()
    {
        if (CurrentUser != null)
        {
            using var db = new AppDbContext();
            AuditLogger.Write(db, CurrentUser.Id, "auth.logout", "Users", CurrentUser.Id, "Logout manual");
        }
        Session.Stop();
        CurrentUser = null;
    }

    public bool ChangePassword(int userId, string currentPassword, string newPassword)
    {
        if (string.IsNullOrWhiteSpace(newPassword) || newPassword.Length < 6)
            return false;

        using var db = new AppDbContext();
        var user = db.Users.Find(userId);
        if (user == null) return false;
        if (!PasswordHasher.Verify(currentPassword, user.PasswordHash)) return false;

        user.PasswordHash = PasswordHasher.Hash(newPassword);
        user.MustChangePassword = false;   // troca concluída → não exigir de novo
        db.SaveChanges();
        AuditLogger.Write(db, userId, "auth.password.changed", "Users", userId, "Senha alterada pelo proprio usuario");

        // Mantém a sessão em memória coerente com o banco.
        if (CurrentUser != null && CurrentUser.Id == userId)
        {
            CurrentUser.PasswordHash = user.PasswordHash;
            CurrentUser.MustChangePassword = false;
        }
        return true;
    }

    public bool CanManageUsers => CurrentUser?.Role == UserRole.Admin;
    public bool CanEdit => CurrentUser?.Role != UserRole.Visualizador;
    public bool CanValidate => CurrentUser?.Role == UserRole.Admin || CurrentUser?.Role == UserRole.Operador;
}

public enum LoginStatus
{
    Success,
    InvalidCredentials,
    Throttled
}

public sealed record LoginOutcome(
    LoginStatus Status,
    User? User,
    int RemainingAttempts,
    TimeSpan BlockedFor)
{
    public bool IsSuccess => Status == LoginStatus.Success;

    public static LoginOutcome Success(User user) =>
        new(LoginStatus.Success, user, int.MaxValue, TimeSpan.Zero);

    public static LoginOutcome InvalidCredentials(int remaining = -1) =>
        new(LoginStatus.InvalidCredentials, null, remaining, TimeSpan.Zero);

    public static LoginOutcome Throttled(TimeSpan duration) =>
        new(LoginStatus.Throttled, null, 0, duration);
}
