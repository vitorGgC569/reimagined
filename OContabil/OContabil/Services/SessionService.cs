using System.Windows;
using System.Windows.Threading;
using OContabil.Models;

namespace OContabil.Services;

/// <summary>
/// Controla o tempo de inatividade da sessão. Após a duração configurada,
/// dispara o evento <see cref="SessionExpired"/> e permite que a janela principal
/// retorne para a tela de login. Atividade do mouse/teclado prorroga a sessão.
/// </summary>
public sealed class SessionService
{
    public event Action? SessionExpired;

    private readonly DispatcherTimer _timer;
    private DateTime _lastActivity;
    private TimeSpan _timeout;
    private User? _user;

    public SessionService()
    {
        _timeout = TimeSpan.FromMinutes(20);
        _lastActivity = DateTime.UtcNow;

        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(30)
        };
        _timer.Tick += OnTimerTick;
    }

    public TimeSpan Timeout
    {
        get => _timeout;
        set => _timeout = value < TimeSpan.FromMinutes(1) ? TimeSpan.FromMinutes(1) : value;
    }

    public bool IsActive => _user != null;
    public User? CurrentUser => _user;

    public void Start(User user, Window window)
    {
        _user = user;
        _lastActivity = DateTime.UtcNow;
        _timer.Start();

        window.PreviewMouseMove += OnUserActivity;
        window.PreviewKeyDown += OnUserActivity;
        window.PreviewMouseDown += OnUserActivity;
        window.Closed += (_, _) => Stop();
    }

    public void Touch() => _lastActivity = DateTime.UtcNow;

    public void Stop()
    {
        _timer.Stop();
        _user = null;
    }

    private void OnUserActivity(object? sender, EventArgs e) => Touch();

    private void OnTimerTick(object? sender, EventArgs e)
    {
        if (_user == null) return;

        if (DateTime.UtcNow - _lastActivity >= _timeout)
        {
            Stop();
            SessionExpired?.Invoke();
        }
    }
}
