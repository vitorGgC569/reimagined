using System.Windows;
using ToastNotifications;
using ToastNotifications.Lifetime;
using ToastNotifications.Messages;
using ToastNotifications.Position;

namespace OContabil.Services;

public static class ToastService
{
    private static Notifier? _notifier;

    public static void Initialize(Window mainWindow)
    {
        _notifier = new Notifier(cfg =>
        {
            cfg.PositionProvider = new WindowPositionProvider(
                parentWindow: mainWindow,
                corner: Corner.BottomRight,
                offsetX: 20,
                offsetY: 20);

            cfg.LifetimeSupervisor = new TimeAndCountBasedLifetimeSupervisor(
                notificationLifetime: TimeSpan.FromSeconds(3),
                maximumNotificationCount: MaximumNotificationCount.FromCount(5));

            cfg.Dispatcher = Application.Current.Dispatcher;
        });
    }

    public static void ShowSuccess(string message)
    {
        Application.Current.Dispatcher.Invoke(() => _notifier?.ShowSuccess(message));
    }

    public static void ShowError(string message)
    {
        Application.Current.Dispatcher.Invoke(() => _notifier?.ShowError(message));
    }

    public static void ShowWarning(string message)
    {
        Application.Current.Dispatcher.Invoke(() => _notifier?.ShowWarning(message));
    }

    public static void ShowInfo(string message)
    {
        Application.Current.Dispatcher.Invoke(() => _notifier?.ShowInformation(message));
    }

    public static void Dispose()
    {
        _notifier?.Dispose();
        _notifier = null;
    }
}
