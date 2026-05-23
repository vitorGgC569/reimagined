using Microsoft.AspNetCore.Mvc;

namespace OContabil.API.Controllers;

[ApiController]
[Route("api/health")]
public sealed class HealthController : ControllerBase
{
    [HttpGet]
    public IActionResult Get() => Ok(new
    {
        status = "ok",
        version = typeof(HealthController).Assembly.GetName().Version?.ToString() ?? "1.0.0",
        timestamp = DateTime.UtcNow
    });
}
