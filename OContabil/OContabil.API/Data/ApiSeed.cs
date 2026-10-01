using OContabil.API.Data.Entities;

namespace OContabil.API.Data;

public static class ApiSeed
{
    public static void Seed(ApiDbContext db)
    {
        if (!db.Users.Any())
        {
            db.Users.Add(new ApiUser
            {
                Username = "admin",
                FullName = "Administrador",
                Email = "admin@ocontabil.local",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("admin"),
                Role = "Admin",
                IsActive = true
            });
            db.SaveChanges();
        }
    }
}
