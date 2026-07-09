using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services.Exports;

internal static class ExportRepository
{
    public static List<Document> Query(AppDbContext db, ExportRequest request)
    {
        IQueryable<Document> q = db.Documents
            .Include(d => d.Client)
            .Include(d => d.ChartOfAccount)
            .Include(d => d.ValidatedBy)
            .Include(d => d.UploadedBy);

        // Seleção explícita tem precedência: exporta exatamente os IDs marcados.
        if (request.DocumentIds is { Count: > 0 } ids)
            q = q.Where(d => ids.Contains(d.Id));

        if (request.OnlyValidated)
            q = q.Where(d => d.Status == DocumentStatus.Validated);

        if (request.StartDate.HasValue)
            q = q.Where(d => d.UploadedAt >= request.StartDate.Value);
        if (request.EndDate.HasValue)
            q = q.Where(d => d.UploadedAt <= request.EndDate.Value);
        if (request.ClientId.HasValue)
            q = q.Where(d => d.ClientId == request.ClientId.Value);
        if (!string.IsNullOrWhiteSpace(request.DocumentType))
            q = q.Where(d => d.DocumentType == request.DocumentType);

        return q.OrderByDescending(d => d.UploadedAt).ToList();
    }
}
