using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using OContabil.API.Data;
using OContabil.API.Data.Entities;
using OContabil.API.Dto;

namespace OContabil.API.Controllers;

[ApiController]
[Route("api/documents")]
[Authorize]
public sealed class DocumentsController : ControllerBase
{
    private readonly ApiDbContext _db;
    private readonly IWebHostEnvironment _env;
    private readonly ILogger<DocumentsController> _log;

    public DocumentsController(ApiDbContext db, IWebHostEnvironment env, ILogger<DocumentsController> log)
    {
        _db = db;
        _env = env;
        _log = log;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<DocumentDto>>> List(
        [FromQuery] int? clientId = null,
        [FromQuery] string? status = null,
        [FromQuery] int page = 1,
        [FromQuery] int pageSize = 50)
    {
        var q = _db.Documents.AsQueryable();
        if (clientId.HasValue) q = q.Where(d => d.ClientId == clientId.Value);
        if (!string.IsNullOrEmpty(status)) q = q.Where(d => d.Status == status);

        var total = await q.CountAsync();
        var items = await q.OrderByDescending(d => d.UploadedAt)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .ToListAsync();
        Response.Headers["X-Total-Count"] = total.ToString();
        return Ok(items.Select(DocumentDto.FromEntity));
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<DocumentDto>> Get(int id)
    {
        var d = await _db.Documents.FindAsync(id);
        return d == null ? NotFound() : Ok(DocumentDto.FromEntity(d));
    }

    [HttpPost("upload")]
    [Authorize(Roles = "Admin,Operator")]
    [RequestSizeLimit(50 * 1024 * 1024)] // 50 MB
    public async Task<ActionResult<DocumentDto>> Upload(
        [FromForm] IFormFile file,
        [FromForm] int clientId,
        [FromForm] string? documentType)
    {
        if (file == null || file.Length == 0)
            return BadRequest(new { message = "Arquivo vazio." });

        var client = await _db.Clients.FindAsync(clientId);
        if (client == null)
            return BadRequest(new { message = "Cliente inexistente." });

        var uploadsDir = Path.Combine(_env.ContentRootPath, "uploads");
        Directory.CreateDirectory(uploadsDir);
        var safeName = $"{Guid.NewGuid():N}_{Path.GetFileName(file.FileName)}";
        var fullPath = Path.Combine(uploadsDir, safeName);
        await using (var stream = System.IO.File.Create(fullPath))
            await file.CopyToAsync(stream);

        var userId = int.TryParse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value
                                  ?? User.FindFirst("sub")?.Value, out var uid) ? uid : 0;

        var doc = new ApiDocument
        {
            Filename = file.FileName,
            DocumentType = documentType ?? "Outro",
            Status = "Pending",
            FileSizeBytes = file.Length,
            UploadedAt = DateTime.UtcNow,
            ClientId = clientId,
            UploadedByUserId = userId
        };
        _db.Documents.Add(doc);
        await _db.SaveChangesAsync();
        _log.LogInformation("Documento {Id} ({File}) recebido", doc.Id, doc.Filename);
        return CreatedAtAction(nameof(Get), new { id = doc.Id }, DocumentDto.FromEntity(doc));
    }

    [HttpPut("{id:int}/status")]
    [Authorize(Roles = "Admin,Operator")]
    public async Task<IActionResult> UpdateStatus(int id, StatusUpdateRequest req)
    {
        var doc = await _db.Documents.FindAsync(id);
        if (doc == null) return NotFound();
        doc.Status = req.Status;
        doc.ProcessedAt = DateTime.UtcNow;
        if (!string.IsNullOrEmpty(req.ExtractedJson))
            doc.ExtractedJson = req.ExtractedJson;
        await _db.SaveChangesAsync();
        return NoContent();
    }
}
