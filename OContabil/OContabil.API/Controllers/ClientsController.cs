using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using OContabil.API.Data;
using OContabil.API.Data.Entities;
using OContabil.API.Dto;

namespace OContabil.API.Controllers;

[ApiController]
[Route("api/clients")]
[Authorize]
public sealed class ClientsController : ControllerBase
{
    private readonly ApiDbContext _db;
    public ClientsController(ApiDbContext db) { _db = db; }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<ClientDto>>> List()
    {
        var clients = await _db.Clients.OrderBy(c => c.Name).ToListAsync();
        return Ok(clients.Select(ClientDto.FromEntity));
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<ClientDto>> Get(int id)
    {
        var c = await _db.Clients.FindAsync(id);
        return c == null ? NotFound() : Ok(ClientDto.FromEntity(c));
    }

    [HttpPost]
    [Authorize(Roles = "Admin,Operator")]
    public async Task<ActionResult<ClientDto>> Create(ClientCreateRequest req)
    {
        if (string.IsNullOrWhiteSpace(req.Name) || string.IsNullOrWhiteSpace(req.Cnpj))
            return BadRequest(new { message = "Nome e CNPJ são obrigatórios." });

        if (await _db.Clients.AnyAsync(c => c.Cnpj == req.Cnpj))
            return Conflict(new { message = "CNPJ já cadastrado." });

        var entity = new ApiClient
        {
            Name = req.Name,
            Cnpj = req.Cnpj,
            Email = req.Email,
            Phone = req.Phone,
            TaxRegime = req.TaxRegime,
            IsActive = true,
            CreatedAt = DateTime.UtcNow
        };
        _db.Clients.Add(entity);
        await _db.SaveChangesAsync();
        return CreatedAtAction(nameof(Get), new { id = entity.Id }, ClientDto.FromEntity(entity));
    }

    [HttpPut("{id:int}")]
    [Authorize(Roles = "Admin,Operator")]
    public async Task<IActionResult> Update(int id, ClientCreateRequest req)
    {
        var c = await _db.Clients.FindAsync(id);
        if (c == null) return NotFound();

        c.Name = req.Name;
        c.Email = req.Email;
        c.Phone = req.Phone;
        c.TaxRegime = req.TaxRegime;
        await _db.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Delete(int id)
    {
        var c = await _db.Clients.FindAsync(id);
        if (c == null) return NotFound();
        c.IsActive = false;
        await _db.SaveChangesAsync();
        return NoContent();
    }
}
