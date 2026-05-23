using Microsoft.AspNetCore.Mvc;
using OContabil.API.Dto;
using OContabil.API.Services;

namespace OContabil.API.Controllers;

[ApiController]
[Route("api/auth")]
public sealed class AuthController : ControllerBase
{
    private readonly AuthService _auth;
    private readonly JwtTokenService _jwt;
    private readonly IConfiguration _config;

    public AuthController(AuthService auth, JwtTokenService jwt, IConfiguration config)
    {
        _auth = auth;
        _jwt = jwt;
        _config = config;
    }

    [HttpPost("login")]
    public async Task<ActionResult<LoginResponse>> Login(LoginRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Username) || string.IsNullOrWhiteSpace(request.Password))
            return BadRequest(new { message = "Usuário e senha são obrigatórios." });

        var user = await _auth.AuthenticateAsync(request.Username, request.Password);
        if (user == null)
            return Unauthorized(new { message = "Credenciais inválidas." });

        var hours = _config.GetValue<int>("Jwt:ExpirationHours", 8);
        var token = _jwt.CreateToken(user, TimeSpan.FromHours(hours));

        return Ok(new LoginResponse(
            Token: token,
            UserId: user.Id,
            Username: user.Username,
            FullName: user.FullName,
            Role: user.Role,
            ExpiresIn: hours * 3600));
    }
}
