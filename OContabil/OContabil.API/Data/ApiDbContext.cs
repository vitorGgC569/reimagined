using Microsoft.EntityFrameworkCore;
using OContabil.API.Data.Entities;

namespace OContabil.API.Data;

public sealed class ApiDbContext : DbContext
{
    public DbSet<ApiUser> Users => Set<ApiUser>();
    public DbSet<ApiClient> Clients => Set<ApiClient>();
    public DbSet<ApiDocument> Documents => Set<ApiDocument>();

    public ApiDbContext(DbContextOptions<ApiDbContext> opt) : base(opt) { }

    protected override void OnModelCreating(ModelBuilder model)
    {
        model.Entity<ApiUser>().HasIndex(u => u.Username).IsUnique();
        model.Entity<ApiClient>().HasIndex(c => c.Cnpj).IsUnique();
    }
}
