using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore;
using OContabil.Data;
using OContabil.Models;

namespace OContabil.Services;

// Bridge between the embedded web design (WebView2) and the real C# services.
// The web UI posts {id, action, payload}; WebShellWindow routes the action here
// and posts the JSON result back.  Each action maps to existing services so the
// design becomes functional without duplicating business logic.
//
// Result convention: { ok: true, data: <...> } or { ok: false, error: "msg" }.
// Dispatch runs on the WPF UI thread (WebMessageReceived), so native dialogs
// (Open/SaveFileDialog) are safe to show from here.
public sealed class WebBridge
{
    private readonly AuthService _auth;

    public WebBridge(AuthService auth) => _auth = auth;

    public object Dispatch(string action, JsonElement payload)
    {
        switch (action)
        {
            case "login": return Login(payload);
            case "session": return Session();
            case "logout": _auth.Logout(); return Ok(new { });

            case "clients.list": return ClientsList();

            case "documents.list": return DocumentsList(payload);
            case "documents.upload": return DocumentsUpload(payload);
            case "documents.reanalyze": return DocumentsReanalyze(payload);
            case "documents.delete": return DocumentsDelete(payload);
            case "documents.validate": return DocumentsSetStatus(payload, validate: true);
            case "documents.reject": return DocumentsSetStatus(payload, validate: false);

            case "exports.run": return ExportsRun(payload);

            case "backup.export": return BackupExport(payload);
            case "backup.restore": return BackupRestore(payload);

            case "organize.plan": return OrganizePlan(payload);
            case "organize.run": return OrganizeRun(payload);

            case "schemas.list": return SchemasList();
            case "schemas.create": return SchemasCreate(payload);
            case "schemas.update": return SchemasUpdate(payload);
            case "users.list": return UsersList();
            case "users.create": return UsersCreate(payload);
            case "audit.list": return AuditList();
            case "dashboard.metrics": return DashboardMetrics();

            default: return Err("Ação desconhecida: " + action);
        }
    }

    // ── Auth ──
    private object Login(JsonElement p)
    {
        var user = Str(p, "user");
        var pass = Str(p, "pass");
        if (string.IsNullOrWhiteSpace(user) || string.IsNullOrEmpty(pass))
            return Err("Preencha usuário e senha.");

        var outcome = _auth.Login(user, pass);
        if (outcome.IsSuccess && outcome.User != null)
            return Ok(UserDto(outcome.User));

        if (outcome.Status == LoginStatus.Throttled)
        {
            var min = Math.Max(1, (int)Math.Ceiling(outcome.BlockedFor.TotalMinutes));
            return Err($"Excesso de tentativas. Tente novamente em {min} min.");
        }
        var msg = outcome.RemainingAttempts is > 0 and < 5
            ? $"Usuário ou senha incorretos. ({outcome.RemainingAttempts} tentativa(s) restante(s))"
            : "Usuário ou senha incorretos.";
        return Err(msg);
    }

    private object Session() =>
        _auth.IsLoggedIn && _auth.CurrentUser != null ? Ok(UserDto(_auth.CurrentUser)) : Err("sem sessão");

    private static object UserDto(Models.User u) => new
    {
        id = u.Id,
        nome = u.FullName,
        usuario = u.Username,
        papel = u.Role.ToString(),
        canManageUsers = u.Role == Models.UserRole.Admin,
    };

    // ── Clients ──
    private object ClientsList()
    {
        using var db = new AppDbContext();
        var rows = db.Clients.OrderBy(c => c.Name).Select(c => new
        {
            id = c.Id,
            razaoSocial = c.Name,
            cnpj = c.Cnpj,
            regime = c.TaxRegime,
            ativo = c.IsActive,
            volume = c.DocumentCount,
        }).ToList();
        return Ok(rows);
    }

    // ── Documents (list / upload / reprocess / status) ──
    private object DocumentsList(JsonElement p)
    {
        using var db = new AppDbContext();
        var docs = db.Documents
            .Include(d => d.Client)
            .Include(d => d.ValidatedBy)
            .OrderByDescending(d => d.Id)
            .Take(500)
            .ToList();

        var now = DateTime.Now;
        var rows = docs.Select(d => DocDto(d, now)).ToList();
        return Ok(rows);
    }

    private object DocDto(Document d, DateTime now)
    {
        var json = d.ExtractedJson;
        int? confianca = d.ConfidenceScore.HasValue ? (int)Math.Round(d.ConfidenceScore.Value * 100) : (int?)null;

        int? diasParado = null;
        if (d.Status is DocumentStatus.Pending or DocumentStatus.ReadyForReview)
        {
            var days = (int)(now - d.UploadedAt).TotalDays;
            if (days >= 3) diasParado = days;
        }

        return new
        {
            id = d.Id,
            tipo = string.IsNullOrEmpty(d.DocumentType) ? "Documento" : d.DocumentType,
            clienteId = d.ClientId,
            cliente = d.Client?.Name ?? "—",
            emitente = FindField(json, "nome_emitente", "razao_social_emitente", "beneficiario", "emitente", "empregador") ?? "",
            numero = FindField(json, "numero_nota", "numero", "numero_referencia") ?? d.Id.ToString(),
            serie = FindField(json, "serie") ?? "",
            chave = FindField(json, "chave_acesso", "chave") ?? "",
            valor = d.ValorTotal ?? ParseBrl(FindField(json, "valor_total", "valor_total_nota", "valor_principal", "salario_liquido", "valor")),
            data = d.UploadedAt.ToString("o"),
            dataStr = d.UploadedAt.ToString("dd/MM/yyyy"),
            dataHora = d.UploadedAt.ToString("dd/MM/yyyy HH:mm"),
            status = WebStatus(d.Status),
            confianca,
            diasParado,
            revisor = d.ValidatedBy?.FullName,
            arquivo = d.Filename,
        };
    }

    private object DocumentsUpload(JsonElement p)
    {
        if (!_auth.IsLoggedIn || _auth.CurrentUser == null) return Err("Sessão expirada. Faça login novamente.");

        int clientId = Int(p, "clienteId");
        using (var probe = new AppDbContext())
        {
            if (clientId <= 0)
                clientId = probe.Clients.Where(c => c.IsActive).OrderBy(c => c.Id).Select(c => c.Id).FirstOrDefault();
            if (clientId <= 0) return Err("Cadastre um cliente antes de importar documentos.");
        }

        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Multiselect = true,
            Title = "Selecionar documentos para extração",
            Filter = "Documentos (*.pdf;*.png;*.jpg;*.jpeg;*.xml;*.ofx;*.txt)|*.pdf;*.png;*.jpg;*.jpeg;*.xml;*.ofx;*.txt|Todos os arquivos (*.*)|*.*",
        };
        if (dlg.ShowDialog() != true) return Ok(new { enqueued = 0, canceled = true });

        int enqueued = 0, rejected = 0;
        var ids = new List<int>();
        const long maxBytes = 50L * 1024 * 1024; // 50 MB/arquivo (anti-DoS/arquivo gigante)
        string[] allowed = { ".pdf", ".png", ".jpg", ".jpeg", ".xml", ".ofx", ".txt", ".csv" };

        foreach (var filePath in dlg.FileNames)
        {
            try
            {
                var fi = new FileInfo(filePath);
                // Validação de entrada: extensão na whitelist + tamanho são (defesa
                // mesmo o diálogo já filtrando — nunca confiar só no cliente).
                if (!fi.Exists || !allowed.Contains(fi.Extension.ToLowerInvariant()) || fi.Length <= 0 || fi.Length > maxBytes)
                {
                    rejected++;
                    SafeLog.Warn("upload.reject", $"arquivo rejeitado (tipo/tamanho): {fi.Name}");
                    continue;
                }

                using var db = new AppDbContext();
                var doc = new Document
                {
                    Filename = fi.Name,
                    FilePath = fi.FullName,
                    DocumentType = GuessDocType(fi.Name),
                    Status = DocumentStatus.Pending,
                    FileSizeBytes = fi.Length,
                    UploadedAt = DateTime.Now,
                    ClientId = clientId,
                    UploadedByUserId = _auth.CurrentUser.Id,
                };
                db.Documents.Add(doc);
                var cli = db.Clients.Find(clientId);
                if (cli != null) cli.DocumentCount++;
                db.SaveChanges();

                DocumentProcessingQueue.Instance.Enqueue(doc.Id, filePath, doc.DocumentType);
                ids.Add(doc.Id);
                enqueued++;
            }
            catch (Exception ex)
            {
                // Um arquivo problemático não derruba o lote inteiro (resiliência).
                SafeLog.Error("upload.item", ex);
                rejected++;
            }
        }
        return Ok(new { enqueued, rejected, ids });
    }

    private object DocumentsReanalyze(JsonElement p)
    {
        int id = Int(p, "id");
        using var db = new AppDbContext();
        var doc = db.Documents.Find(id);
        if (doc == null) return Err("Documento não encontrado.");
        if (string.IsNullOrEmpty(doc.FilePath) || !File.Exists(doc.FilePath))
            return Err("Arquivo original não encontrado: " + (doc.FilePath ?? "desconhecido"));
        DocumentProcessingQueue.Instance.Enqueue(doc.Id, doc.FilePath, doc.DocumentType);
        return Ok(new { id });
    }

    private object DocumentsDelete(JsonElement p)
    {
        int id = Int(p, "id");
        using var db = new AppDbContext();
        var doc = db.Documents.Find(id);
        if (doc == null) return Err("Documento não encontrado.");
        var cli = db.Clients.Find(doc.ClientId);
        if (cli != null) cli.DocumentCount = Math.Max(0, cli.DocumentCount - 1);
        db.Documents.Remove(doc);
        db.SaveChanges();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "document.delete", "Documents", id, doc.Filename);
        return Ok(new { id });
    }

    private object DocumentsSetStatus(JsonElement p, bool validate)
    {
        int id = Int(p, "id");
        using var db = new AppDbContext();
        var doc = db.Documents.Include(d => d.Client).FirstOrDefault(d => d.Id == id);
        if (doc == null) return Err("Documento não encontrado.");

        if (validate)
        {
            doc.Status = DocumentStatus.Validated;
            doc.ValidatedAt = DateTime.Now;
            doc.ValidatedByUserId = _auth.CurrentUser?.Id;
            if (doc.Client != null) doc.Client.ValidatedCount++;
        }
        else
        {
            doc.Status = DocumentStatus.Rejected;
            doc.RejectionReason = Str(p, "reason");
            doc.ValidatedByUserId = _auth.CurrentUser?.Id;
        }
        db.SaveChanges();
        AuditLogger.Write(db, _auth.CurrentUser?.Id,
            validate ? "document.validate" : "document.reject", "Documents", id, doc.RejectionReason);
        return Ok(new { id, status = WebStatus(doc.Status) });
    }

    // ── Exports (real CSV/TXT of the extracted data, saved via native dialog) ──
    private object ExportsRun(JsonElement p)
    {
        string fmt = Str(p, "format");
        int clientId = Int(p, "clienteId");
        string tipo = Str(p, "tipo");
        string status = Str(p, "status");

        using var db = new AppDbContext();
        var q = db.Documents.Include(d => d.Client).AsQueryable();
        var ids = IntArray(p, "ids");
        if (ids.Length > 0) q = q.Where(d => ids.Contains(d.Id));
        if (clientId > 0) q = q.Where(d => d.ClientId == clientId);
        if (!string.IsNullOrEmpty(tipo) && tipo != "todos") q = q.Where(d => d.DocumentType == tipo);
        if (status == "aprovado") q = q.Where(d => d.Status == DocumentStatus.Validated);

        var docs = q.OrderByDescending(d => d.Id).ToList();
        if (docs.Count == 0) return Err("Nenhum documento corresponde ao filtro selecionado.");

        // Honest scope: official SPED/Domínio binary layouts are not implemented;
        // every format exports the real extracted data as UTF-8 CSV (opens in Excel).
        bool official = fmt is "sped" or "dominio";
        var sfd = new Microsoft.Win32.SaveFileDialog
        {
            FileName = $"export_{(string.IsNullOrEmpty(fmt) ? "csv" : fmt)}_{DateTime.Now:yyyyMM}.csv",
            Filter = "CSV UTF-8 (*.csv)|*.csv|Todos os arquivos (*.*)|*.*",
            Title = "Salvar exportação",
        };
        if (sfd.ShowDialog() != true) return Ok(new { canceled = true });

        var sb = new StringBuilder();
        sb.AppendLine("id;arquivo;tipo;cliente;cnpj;status;confianca;data_upload;valor;json_extraido");
        foreach (var d in docs)
        {
            var valor = ParseBrl(FindField(d.ExtractedJson, "valor_total", "valor_total_nota", "valor_principal", "valor"));
            sb.Append(d.Id).Append(';')
              .Append(Csv(d.Filename)).Append(';')
              .Append(Csv(d.DocumentType)).Append(';')
              .Append(Csv(d.Client?.Name)).Append(';')
              .Append(Csv(d.Client?.Cnpj)).Append(';')
              .Append(Csv(d.StatusDisplay)).Append(';')
              .Append((d.ConfidenceScore ?? 0).ToString("0.00", CultureInfo.InvariantCulture)).Append(';')
              .Append(d.UploadedAt.ToString("yyyy-MM-dd")).Append(';')
              .Append(valor.ToString("0.00", CultureInfo.InvariantCulture)).Append(';')
              .Append(Csv(d.ExtractedJson))
              .AppendLine();
        }
        File.WriteAllText(sfd.FileName, sb.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "export.run", "Documents", null,
            $"format={fmt}; count={docs.Count}; file={Path.GetFileName(sfd.FileName)}");

        return Ok(new
        {
            path = sfd.FileName,
            count = docs.Count,
            note = official ? $"Layout oficial {fmt.ToUpperInvariant()} ainda não implementado — dados exportados em CSV." : (string?)null,
        });
    }

    // ── Backup / Recuperação (cifrado por SENHA, independente do db.key/DPAPI) ──
    private object BackupExport(JsonElement p)
    {
        if (_auth.CurrentUser?.Role != Models.UserRole.Admin) return Err("Apenas administradores podem exportar backup.");
        var senha = Str(p, "senha");
        if (senha.Length < 6) return Err("Defina uma senha de backup com ao menos 6 caracteres.");

        var sfd = new Microsoft.Win32.SaveFileDialog
        {
            FileName = $"ocontabil_backup_{DateTime.Now:yyyyMMdd_HHmm}.ocbak",
            Filter = "Backup OContabil (*.ocbak)|*.ocbak|Todos os arquivos (*.*)|*.*",
            Title = "Salvar backup cifrado",
        };
        if (sfd.ShowDialog() != true) return Ok(new { canceled = true });

        try { DbBackupService.Export(sfd.FileName, senha); }
        catch (Exception ex) { SafeLog.Error("backup.export", ex); return Err("Não foi possível gerar o backup."); }

        using var db = new AppDbContext();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "backup.export", "Database", null, Path.GetFileName(sfd.FileName));
        return Ok(new { path = sfd.FileName, note = "Guarde o arquivo e a senha em local seguro — sem a senha, o backup é irrecuperável." });
    }

    private object BackupRestore(JsonElement p)
    {
        if (_auth.CurrentUser?.Role != Models.UserRole.Admin) return Err("Apenas administradores podem restaurar backup.");
        var senha = Str(p, "senha");
        if (senha.Length < 6) return Err("Informe a senha do backup.");

        var ofd = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "Backup OContabil (*.ocbak)|*.ocbak|Todos os arquivos (*.*)|*.*",
            Title = "Selecionar backup para restaurar",
            CheckFileExists = true,
        };
        if (ofd.ShowDialog() != true) return Ok(new { canceled = true });

        try { DbBackupService.StageRestore(ofd.FileName, senha); }
        catch (Exception ex) { SafeLog.Error("backup.restore", ex); return Err("Senha incorreta ou arquivo de backup inválido."); }

        using var db = new AppDbContext();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "backup.restore", "Database", null, Path.GetFileName(ofd.FileName));
        return Ok(new { staged = true, note = "Backup validado. Feche e reabra o OContabil para concluir a restauração." });
    }

    // ── Organização (estrutura de pastas determinística, sem IA) ──
    private object OrganizePlan(JsonElement p)
    {
        if (!_auth.IsLoggedIn) return Err("sem sessão");
        int clientId = Int(p, "clienteId");
        var plan = DocumentOrganizerService.BuildPlan(clientId > 0 ? clientId : (int?)null);
        return Ok(plan);
    }

    private object OrganizeRun(JsonElement p)
    {
        if (!_auth.IsLoggedIn) return Err("sem sessão");
        int clientId = Int(p, "clienteId");
        var dlg = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Escolha a pasta raiz para organizar os documentos",
        };
        if (dlg.ShowDialog() != true) return Ok(new { canceled = true });

        var report = DocumentOrganizerService.Organize(dlg.FolderName, clientId > 0 ? clientId : (int?)null);
        using var db = new AppDbContext();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "files.organize", "Documents", null,
            $"root={report.Raiz}; ok={report.Organizados}; skip={report.Ignorados}; err={report.Erros}");
        return Ok(report);
    }

    // ── Schemas ──
    private object SchemasList()
    {
        using var db = new AppDbContext();
        var rows = db.DocumentSchemas.OrderBy(s => s.DocumentType).ToList().Select(s => new
        {
            id = s.Id,
            nome = s.Name,
            tipo = s.DocumentType,
            sistema = s.IsSystem,
            clienteId = s.ClientId,
            campos = ParseSchemaFields(s.SchemaJson),
        }).ToList();
        return Ok(rows);
    }

    private static List<object> ParseSchemaFields(string schemaJson)
    {
        var list = new List<object>();
        try
        {
            using var doc = JsonDocument.Parse(schemaJson);
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return list;
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.Array) continue;
                foreach (var f in prop.Value.EnumerateArray())
                {
                    var raw = f.GetString() ?? "";
                    var parts = raw.Split("::");
                    list.Add(new
                    {
                        nome = parts.Length > 0 ? parts[0] : raw,
                        tipo = parts.Length > 1 ? parts[1] : "str",
                        desc = parts.Length > 2 ? parts[2] : "",
                    });
                }
            }
        }
        catch { }
        return list;
    }

    // ── Users + Audit ──
    private object UsersList()
    {
        if (!_auth.IsLoggedIn) return Err("sem sessão");
        using var db = new AppDbContext();
        var rows = db.Users.OrderBy(u => u.FullName).ToList().Select(u => new
        {
            id = u.Id,
            nome = u.FullName,
            email = string.IsNullOrEmpty(u.Email) ? "—" : u.Email,
            papel = u.RoleDisplay,
            ultimoLogin = u.LastLoginAt?.ToString("dd/MM/yyyy HH:mm") ?? "—",
            status = !u.IsActive ? "bloqueado" : u.LastLoginAt == null ? "convidado" : "ativo",
            tentativas = 0,
            mfa = false,
        }).ToList();
        return Ok(rows);
    }

    private object AuditList()
    {
        using var db = new AppDbContext();
        var logs = db.AuditLogs.OrderByDescending(a => a.Timestamp).Take(25).ToList();
        var names = db.Users.ToDictionary(u => u.Id, u => u.FullName);
        var rows = logs.Select(a => new
        {
            quem = a.UserId.HasValue && names.TryGetValue(a.UserId.Value, out var n) ? n : "Sistema",
            acao = string.IsNullOrEmpty(a.Details) ? a.Action : $"{a.Action} — {a.Details}",
            quando = a.Timestamp.ToLocalTime().ToString("dd/MM/yyyy HH:mm"),
            ip = string.IsNullOrEmpty(a.IpAddress) ? "localhost" : a.IpAddress,
            tipo = AuditTipo(a.Action),
        }).ToList();
        return Ok(rows);
    }

    private static string AuditTipo(string action)
    {
        action = (action ?? "").ToLowerInvariant();
        if (action.Contains("export")) return "export";
        if (action.Contains("login") || action.Contains("auth") || action.Contains("lock")) return "seguranca";
        if (action.Contains("process")) return "sistema";
        if (action.Contains("validate") || action.Contains("reject") || action.Contains("edit")) return "edicao";
        if (action.Contains("delete")) return "seguranca";
        if (action.Contains("schema") || action.Contains("config") || action.Contains("client")) return "config";
        return "sistema";
    }

    // ── Dashboard ──
    private object DashboardMetrics()
    {
        using var db = new AppDbContext();
        var docs = db.Documents.ToList();
        var now = DateTime.Now;
        var today = now.Date;

        var validated = docs.Where(d => d.Status == DocumentStatus.Validated && d.ConfidenceScore.HasValue).ToList();
        double confMedia = validated.Count > 0 ? Math.Round(validated.Average(d => d.ConfidenceScore!.Value) * 100, 1) : 0;

        int parados = docs.Count(d =>
            (d.Status is DocumentStatus.Pending or DocumentStatus.ReadyForReview) &&
            (now - d.UploadedAt).TotalDays >= 3);

        var kpis = new
        {
            processadosHoje = docs.Count(d => d.ProcessedAt.HasValue && d.ProcessedAt.Value.Date == today),
            naFila = DocumentProcessingQueue.Instance.PendingCount,
            aguardandoRevisao = docs.Count(d => d.Status == DocumentStatus.ReadyForReview),
            confiancaMedia = confMedia,
            docsParados = parados,
        };

        var statusDist = new
        {
            aprovado = docs.Count(d => d.Status == DocumentStatus.Validated),
            revisar = docs.Count(d => d.Status == DocumentStatus.ReadyForReview),
            rejeitado = docs.Count(d => d.Status is DocumentStatus.Rejected or DocumentStatus.Error),
            pendente = docs.Count(d => d.Status is DocumentStatus.Pending or DocumentStatus.Processing),
        };

        var topClientes = db.Clients.OrderByDescending(c => c.DocumentCount).Take(6).ToList().Select(c => new
        {
            id = c.Id,
            nome = c.Name,
            fantasia = c.Name,
            regime = c.TaxRegime,
            volume = c.DocumentCount,
        }).ToList();

        // 14-day series from real upload dates (volume + avg confidence per day).
        var confSeries = new List<object>();
        for (int i = 13; i >= 0; i--)
        {
            var day = today.AddDays(-i);
            var dayDocs = docs.Where(d => d.UploadedAt.Date == day).ToList();
            var dayConf = dayDocs.Where(d => d.ConfidenceScore.HasValue).ToList();
            confSeries.Add(new
            {
                dia = day.ToString("dd/MM"),
                vol = dayDocs.Count,
                conf = dayConf.Count > 0 ? Math.Round(dayConf.Average(d => d.ConfidenceScore!.Value) * 100, 1) : confMedia,
            });
        }

        return Ok(new { kpis, statusDist, topClientes, confSeries });
    }

    // ── Usuários (criação — admin) ──
    private object UsersCreate(JsonElement p)
    {
        if (_auth.CurrentUser?.Role != Models.UserRole.Admin) return Err("Apenas administradores podem criar usuários.");
        var nome = Str(p, "nome");
        var usuario = Str(p, "usuario");
        var senha = Str(p, "senha");
        if (string.IsNullOrWhiteSpace(nome) || string.IsNullOrWhiteSpace(usuario) || senha.Length < 6)
            return Err("Informe nome, login e senha (mín. 6 caracteres).");

        using var db = new AppDbContext();
        if (db.Users.Any(u => u.Username == usuario)) return Err("Já existe um usuário com esse login.");

        var role = Str(p, "papel") switch
        {
            "Administrador" => Models.UserRole.Admin,
            "Visualizador" => Models.UserRole.Visualizador,
            _ => Models.UserRole.Operador,
        };
        var user = new Models.User
        {
            Username = usuario, FullName = nome, Email = Str(p, "email"),
            PasswordHash = PasswordHasher.Hash(senha), Role = role,
            IsActive = true, MustChangePassword = true, CreatedAt = DateTime.Now,
        };
        db.Users.Add(user);
        db.SaveChanges();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "user.create", "Users", user.Id, usuario);
        return Ok(new { id = user.Id });
    }

    // ── Schemas (criação/edição — apenas não-sistema) ──
    private object SchemasCreate(JsonElement p)
    {
        if (!_auth.IsLoggedIn) return Err("sem sessão");
        var nome = Str(p, "nome");
        var tipo = Str(p, "tipo");
        if (string.IsNullOrWhiteSpace(nome) || string.IsNullOrWhiteSpace(tipo))
            return Err("Informe nome e tipo do schema.");
        var cid = Int(p, "clienteId");

        using var db = new AppDbContext();
        var s = new DocumentSchema
        {
            Name = nome,
            DocumentType = tipo,
            IsSystem = false,
            ClientId = cid > 0 ? cid : (int?)null,
            SchemaJson = BuildSchemaJson(tipo, StrArray(p, "campos")),
            CreatedAt = DateTime.UtcNow,
        };
        db.DocumentSchemas.Add(s);
        db.SaveChanges();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "schema.create", "DocumentSchemas", s.Id, nome);
        return Ok(new { id = s.Id });
    }

    private object SchemasUpdate(JsonElement p)
    {
        if (!_auth.IsLoggedIn) return Err("sem sessão");
        int id = Int(p, "id");
        using var db = new AppDbContext();
        var s = db.DocumentSchemas.Find(id);
        if (s == null) return Err("Schema não encontrado.");
        if (s.IsSystem) return Err("Schemas do sistema não podem ser editados.");
        s.SchemaJson = BuildSchemaJson(s.DocumentType, StrArray(p, "campos"));
        db.SaveChanges();
        AuditLogger.Write(db, _auth.CurrentUser?.Id, "schema.update", "DocumentSchemas", s.Id, s.Name);
        return Ok(new { id });
    }

    private static string BuildSchemaJson(string tipo, string[] campos)
    {
        var group = string.IsNullOrWhiteSpace(tipo) ? "campos" : tipo.ToLowerInvariant().Replace('-', '_').Replace(' ', '_');
        var fields = campos
            .Select(c => c.Trim())
            .Where(c => c.Length > 0)
            .Select(c => c.Contains("::") ? c : c + "::str::")
            .ToList();
        return JsonSerializer.Serialize(new Dictionary<string, List<string>> { [group] = fields });
    }

    // ── helpers ──
    private static object Ok(object data) => new { ok = true, data };
    private static object Err(string error) => new { ok = false, error };

    private static string WebStatus(DocumentStatus s) => s switch
    {
        DocumentStatus.Validated => "aprovado",
        DocumentStatus.ReadyForReview => "revisar",
        DocumentStatus.Rejected => "rejeitado",
        DocumentStatus.Error => "rejeitado",
        DocumentStatus.Processing => "processando",
        _ => "pendente",
    };

    private static string GuessDocType(string filename)
    {
        var n = filename.ToLowerInvariant();
        if (n.EndsWith(".ofx")) return "ExtratoBancario";
        if (n.Contains("boleto")) return "Boleto";
        if (n.Contains("darf")) return "DARF";
        if (n.Contains("holerite") || n.Contains("folha") || n.Contains("contracheque")) return "Holerite";
        if (n.Contains("danfe")) return "DANFE";
        if (n.Contains("extrato")) return "ExtratoBancario";
        return "NF-e";
    }

    // Recursively find the first string/number value whose key contains any part.
    private static string? FindField(string? json, params string[] keyParts)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try { using var doc = JsonDocument.Parse(json); return FindIn(doc.RootElement, keyParts); }
        catch { return null; }
    }

    private static string? FindIn(JsonElement el, string[] keyParts)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var p in el.EnumerateObject())
                {
                    if (!keyParts.Any(k => p.Name.Contains(k, StringComparison.OrdinalIgnoreCase))) continue;
                    if (p.Value.ValueKind is JsonValueKind.String or JsonValueKind.Number)
                        return p.Value.ToString();
                    // extração no formato {campo: {text, confidence}}
                    if (p.Value.ValueKind == JsonValueKind.Object &&
                        p.Value.TryGetProperty("text", out var tv) &&
                        tv.ValueKind is JsonValueKind.String or JsonValueKind.Number)
                        return tv.ToString();
                }
                foreach (var p in el.EnumerateObject())
                {
                    var r = FindIn(p.Value, keyParts);
                    if (r != null) return r;
                }
                return null;
            case JsonValueKind.Array:
                foreach (var it in el.EnumerateArray())
                {
                    var r = FindIn(it, keyParts);
                    if (r != null) return r;
                }
                return null;
            default:
                return null;
        }
    }

    private static double ParseBrl(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return 0;
        var t = Regex.Replace(s, @"[^\d,.\-]", "");
        if (string.IsNullOrEmpty(t)) return 0;
        if (t.Contains(',')) t = t.Replace(".", "").Replace(",", ".");
        return double.TryParse(t, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0;
    }

    private static string Csv(string? s) => SecureCsv.Cell(s);

    private static string Str(JsonElement p, string key) =>
        p.ValueKind == JsonValueKind.Object && p.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() ?? "" : "";

    private static int Int(JsonElement p, string key)
    {
        if (p.ValueKind != JsonValueKind.Object || !p.TryGetProperty(key, out var v)) return 0;
        if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var n)) return n;
        if (v.ValueKind == JsonValueKind.String && int.TryParse(v.GetString(), out var s)) return s;
        return 0;
    }

    private static int[] IntArray(JsonElement p, string key)
    {
        if (p.ValueKind != JsonValueKind.Object || !p.TryGetProperty(key, out var v) || v.ValueKind != JsonValueKind.Array)
            return Array.Empty<int>();
        var list = new List<int>();
        foreach (var e in v.EnumerateArray())
        {
            if (e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out var n)) list.Add(n);
            else if (e.ValueKind == JsonValueKind.String && int.TryParse(e.GetString(), out var s)) list.Add(s);
        }
        return list.ToArray();
    }

    private static string[] StrArray(JsonElement p, string key)
    {
        if (p.ValueKind != JsonValueKind.Object || !p.TryGetProperty(key, out var v) || v.ValueKind != JsonValueKind.Array)
            return Array.Empty<string>();
        var list = new List<string>();
        foreach (var e in v.EnumerateArray())
            if (e.ValueKind == JsonValueKind.String) list.Add(e.GetString() ?? "");
        return list.ToArray();
    }
}
