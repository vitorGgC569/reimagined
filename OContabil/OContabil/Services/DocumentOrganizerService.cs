using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services;

/// <summary>
/// Organizador determinístico de arquivos (sem IA): estrutura
///   Raiz / {Empresa (CNPJ)} / {Ano} / {MM-Mês} / {Tipo} / arquivo
/// a partir dos metadados já extraídos — empresa via cliente, data via extração
/// (ou data de upload), tipo via classificação. É só I/O + template de caminho:
/// rápido, exato, idempotente (deduplica por conteúdo), copia (preserva o
/// original) e totalmente guardado (um arquivo problemático não derruba o lote).
/// </summary>
public static class DocumentOrganizerService
{
    private static readonly string[] Meses =
        { "", "01-Janeiro", "02-Fevereiro", "03-Marco", "04-Abril", "05-Maio", "06-Junho",
          "07-Julho", "08-Agosto", "09-Setembro", "10-Outubro", "11-Novembro", "12-Dezembro" };

    public sealed record Plan(int Total, int ComArquivo, int SemArquivo, List<PlanNode> Arvore);
    public sealed record PlanNode(string Nome, int Total, List<PlanNode>? Filhos);
    public sealed record Report(int Organizados, int Ignorados, int Erros, string Raiz, List<string> Mensagens);

    /// <summary>Pré-visualização (não toca o disco): conta por Empresa → Ano → Mês → Tipo.</summary>
    public static Plan BuildPlan(int? clientId = null)
    {
        var docs = LoadDocs(clientId);
        int comArquivo = docs.Count(d => HasFile(d.doc));

        var arvore = docs
            .GroupBy(d => d.empresa).OrderBy(g => g.Key)
            .Select(emp => new PlanNode(emp.Key, emp.Count(),
                emp.GroupBy(d => d.ano).OrderByDescending(g => g.Key)
                   .Select(ano => new PlanNode(ano.Key, ano.Count(),
                       ano.GroupBy(d => d.mes).OrderBy(g => g.Key)
                          .Select(mes => new PlanNode(MesLabel(mes.Key), mes.Count(),
                              mes.GroupBy(d => d.tipo).OrderBy(g => g.Key)
                                 .Select(t => new PlanNode(t.Key, t.Count(), null)).ToList()))
                          .ToList()))
                   .ToList()))
            .ToList();

        return new Plan(docs.Count, comArquivo, docs.Count - comArquivo, arvore);
    }

    /// <summary>Executa a organização (copia os arquivos para a estrutura sob <paramref name="root"/>).</summary>
    public static Report Organize(string root, int? clientId = null)
    {
        var msgs = new List<string>();
        int ok = 0, skip = 0, err = 0;

        if (string.IsNullOrWhiteSpace(root))
            return new Report(0, 0, 0, root ?? "", new List<string> { "Pasta raiz inválida." });

        try { Directory.CreateDirectory(root); }
        catch (Exception ex) { SafeLog.Error("organize.root", ex); return new Report(0, 0, 0, root, new List<string> { "Não foi possível criar a pasta raiz." }); }

        // Resiliência: não inicia sem espaço mínimo em disco.
        try
        {
            var drive = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(root)) ?? "C:\\");
            if (drive.IsReady && drive.AvailableFreeSpace < 50L * 1024 * 1024)
                return new Report(0, 0, 0, root, new List<string> { "Espaço em disco insuficiente (< 50 MB)." });
        }
        catch { }

        foreach (var d in LoadDocs(clientId))
        {
            if (!HasFile(d.doc)) { skip++; continue; }
            try
            {
                // Confinamento na raiz (anti path-traversal): segmentos vêm de
                // dados extraídos/nome de arquivo de terceiros — nunca confiáveis.
                var dir = SecurePath.CombineInside(root, d.empresa, d.ano, d.mes, d.tipo);
                if (dir == null) { err++; if (msgs.Count < 20) msgs.Add($"{d.doc.Filename}: caminho rejeitado (traversal)"); continue; }
                Directory.CreateDirectory(dir);

                var name = SecurePath.SanitizeSegment(Path.GetFileName(d.doc.FilePath!));
                var target = Path.Combine(dir, name);
                if (!SecurePath.IsInside(root, target)) { err++; continue; }

                if (File.Exists(target))
                {
                    // Dedup idempotente: mesmo tamanho = mesmo arquivo → não recopia.
                    if (new FileInfo(d.doc.FilePath!).Length == new FileInfo(target).Length) { skip++; continue; }
                    var stem = Path.GetFileNameWithoutExtension(name);
                    var ext = Path.GetExtension(name);
                    target = Path.Combine(dir, $"{stem}_{d.doc.Id}{ext}");
                    if (File.Exists(target)) { skip++; continue; }
                }

                File.Copy(d.doc.FilePath!, target, overwrite: false);
                ok++;
            }
            catch (Exception ex)
            {
                err++;
                if (msgs.Count < 20) msgs.Add($"{d.doc.Filename}: {ex.Message}");
            }
        }

        return new Report(ok, skip, err, root, msgs);
    }

    // ── internals ──
    private static List<(Document doc, string empresa, string ano, string mes, string tipo)> LoadDocs(int? clientId)
    {
        using var db = new AppDbContext();
        var q = db.Documents.Include(x => x.Client).AsQueryable();
        if (clientId is > 0) q = q.Where(x => x.ClientId == clientId);

        return q.OrderBy(x => x.ClientId).ToList().Select(doc =>
        {
            var empresa = EmpresaFolder(doc.Client);
            var date = DocDate(doc);
            var ano = date.Year.ToString();
            var mes = (date.Month >= 1 && date.Month <= 12) ? Meses[date.Month] : "00-Indefinido";
            var tipo = string.IsNullOrWhiteSpace(doc.DocumentType) ? "Outros" : doc.DocumentType;
            return (doc, empresa, ano, mes, tipo);
        }).ToList();
    }

    private static bool HasFile(Document d) => !string.IsNullOrEmpty(d.FilePath) && File.Exists(d.FilePath);

    private static string EmpresaFolder(Client? c)
    {
        if (c == null) return "Sem cliente";
        var name = string.IsNullOrWhiteSpace(c.Name) ? "Cliente" : c.Name;
        var cnpj = string.IsNullOrWhiteSpace(c.Cnpj) ? "" : " (" + Regex.Replace(c.Cnpj, @"[^\d]", "") + ")";
        return name + cnpj;
    }

    private static DateTime DocDate(Document d)
    {
        var s = FindDate(d.ExtractedJson);
        if (s != null && DateTime.TryParseExact(s.Trim(),
                new[] { "dd/MM/yyyy", "d/M/yyyy", "MM/yyyy", "M/yyyy", "yyyy-MM-dd" },
                CultureInfo.InvariantCulture, DateTimeStyles.None, out var dt))
            return dt;
        return d.UploadedAt;
    }

    private static string MesLabel(string mes) => mes.Length > 3 ? mes.Substring(3) : mes;

    private static string? FindDate(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            using var doc = JsonDocument.Parse(json);
            foreach (var key in new[] { "data_emissao", "data_vencimento", "vencimento", "competencia", "data" })
            {
                var v = FindKey(doc.RootElement, key);
                if (!string.IsNullOrWhiteSpace(v)) return v;
            }
        }
        catch { }
        return null;
    }

    private static string? FindKey(JsonElement el, string keyPart)
    {
        if (el.ValueKind != JsonValueKind.Object) return null;
        foreach (var p in el.EnumerateObject())
        {
            if (!p.Name.Contains(keyPart, StringComparison.OrdinalIgnoreCase)) continue;
            if (p.Value.ValueKind is JsonValueKind.String or JsonValueKind.Number) return p.Value.ToString();
            if (p.Value.ValueKind == JsonValueKind.Object && p.Value.TryGetProperty("text", out var t)) return t.ToString();
        }
        foreach (var p in el.EnumerateObject()) { var r = FindKey(p.Value, keyPart); if (r != null) return r; }
        return null;
    }

}
