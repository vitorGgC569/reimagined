using Microsoft.EntityFrameworkCore;
using OContabil.API.Data;
using OContabil.API.Data.Entities;

namespace OContabil.API.Services;

public sealed class AuthService
{
    private readonly ApiDbContext _db;
    public AuthService(ApiDbContext db) { _db = db; }

    public async Task<ApiUser?> AuthenticateAsync(string username, string password)
    {
        var user = await _db.Users.FirstOrDefaultAsync(u =>
            u.Username == username && u.IsActive);
        if (user == null) return null;

        if (!BCrypt.Net.BCrypt.Verify(password, user.PasswordHash))
            return null;

        user.LastLoginAt = DateTime.UtcNow;
        await _db.SaveChangesAsync();
        return user;
    }
}
