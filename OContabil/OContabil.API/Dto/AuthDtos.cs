using OContabil.API.Data.Entities;

namespace OContabil.API.Dto;

public sealed record LoginRequest(string Username, string Password);

public sealed record LoginResponse(
    string Token,
    int UserId,
    string Username,
    string FullName,
    string Role,
    int ExpiresIn);

public sealed record ClientCreateRequest(
    string Name,
    string Cnpj,
    string? Email,
    string? Phone,
    string? TaxRegime);

public sealed record ClientDto(
    int Id, string Name, string Cnpj, string? Email, string? Phone, string? TaxRegime, bool IsActive)
{
    public static ClientDto FromEntity(ApiClient c) =>
        new(c.Id, c.Name, c.Cnpj, c.Email, c.Phone, c.TaxRegime, c.IsActive);
}

public sealed record DocumentDto(
    int Id, string Filename, string DocumentType, string Status,
    double? ConfidenceScore, long FileSizeBytes,
    DateTime UploadedAt, DateTime? ProcessedAt,
    int ClientId, string? ExtractedJson)
{
    public static DocumentDto FromEntity(ApiDocument d) =>
        new(d.Id, d.Filename, d.DocumentType, d.Status, d.ConfidenceScore,
            d.FileSizeBytes, d.UploadedAt, d.ProcessedAt, d.ClientId, d.ExtractedJson);
}

public sealed record StatusUpdateRequest(string Status, string? ExtractedJson);
