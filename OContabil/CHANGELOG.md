# Changelog

Todas as mudanças notáveis serão documentadas neste arquivo. O formato segue
[Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e este projeto adere
ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [1.0.0] — 2026-05-12

### Adicionado — Fase 1: Fundação segura
- **PBKDF2-SHA256** com salt aleatório e 120k iterações; auto-upgrade de hashes
  SHA-256 legados na próxima autenticação válida.
- **Rate limiting** de login: 5 tentativas em 10 minutos → bloqueio de 15 min.
- **Timeout de sessão** por inatividade (configurável, padrão 20 min) com hook
  em mouse/teclado da janela WPF.
- **AuditLogs**: tabela com registro de toda ação relevante (login, processamento,
  revisão, exportação, exclusão). Best-effort, nunca lança exceções.
- **Backup automático** do SQLite com rotação de N arquivos (configurável).
- **SchemaManager**: upgrades SQL idempotentes (v1–v4) — substituem migrações
  EF Core para o cenário desktop com `EnsureCreated`.

### Adicionado — Fase 2: Motor de IA
- **ONNX Runtime nativo** com `Microsoft.ML.Tokenizers` BertTokenizer — elimina
  dependência obrigatória do processo Python para extração.
- **Cadeia de engines configurável**: ONNX → Python → Regex, com timeout por
  engine, retries e fallback gracioso.
- **RegexExtractionService** offline para NF-e, DANFE, Boleto, Holerite,
  ExtratoBancario, DARF — funciona sem Python ou modelo.

### Adicionado — Fase 3: Funcionalidades de negócio
- **5 formatos de exportação**:
  - **Excel** (ClosedXML) — 3 abas (Resumo, Documentos com auto-filter, Por Cliente)
  - **CSV** UTF-8 BOM separado por `;` (Excel pt-BR)
  - **SPED Fiscal** conforme ATO COTEPE/ICMS Nº 44/2018 (blocos 0/C/9)
  - **Domínio Sistemas** pipe-delimitado
  - **PDF de Conferência** (QuestPDF) com cabeçalho da empresa
- **Dashboard com LiveChartsCore**: pizza de status, top 5 clientes, série
  temporal de confiança da IA (14 dias), banner de documentos parados.
- **DocumentReviewDialog** completo com 3 abas:
  - **Revisão**: viewer original + OCR + status com motivo de rejeição obrigatório
  - **Campos Extraídos**: edição campo a campo de JSON estruturado
  - **Histórico**: trilha de auditoria com diff JSON e usuário responsável
- **Schemas IA customizáveis**: cadastro de modelos de extração por tipo de
  documento e por cliente, com 6 schemas built-in (NF-e, Boleto, Holerite,
  ExtratoBancário, DARF, DANFE).
- **OfxParserService**: parsing OFX 1.x (SGML) e 2.x (XML) — extratos Itaú,
  Bradesco, Santander, Caixa.

### Adicionado — Fase 4: Distribuição
- **Inno Setup script** (`installer/OContabil.iss`) com checagem de .NET runtime.
- **FirstRunWizard** em 3 etapas: empresa → admin → dependências (Python/Tesseract).
- Suporte a `embedded Python` opcional via `installer/python/`.

### Adicionado — Fase 5: SaaS / API
- **OContabil.API** (ASP.NET Core 8): endpoints `/auth/login`, `/clients`,
  `/documents` com upload multipart, atualizações de status e listagem paginada.
- **JWT** (BCrypt para hash de senhas; chave configurável; expiração 8h).
- **OpenAPI/Swagger** habilitado em ambiente Development.
- **`gliner2-service` FastAPI** dockerizado: endpoint `/extract` e `/extract-file`
  com OCR via PyMuPDF/Tesseract.
- **docker-compose.yml**: stack completa API + GLiNER + Nginx reverse proxy.
- **api-client.js** no frontend SaaS substituindo `mock-data.js` por chamadas
  reais (login, upload, polling).

### Adicionado — Fase 6: Profissionalismo
- **Serilog** estruturado: logs diários em `%LocalAppData%\OContabil\logs\`,
  retenção 14 dias.
- **Mapeamento de mensagens amigáveis** (UnauthorizedAccess, FileNotFound,
  IOException, HttpRequestException) em vez de stack traces.
- **TelemetryService opt-in**: contadores locais (processados, validados,
  falhas, exports) — nenhum dado deixa a máquina.
- **GitHub Actions**: `.github/workflows/dotnet.yml` (build + test + coverage)
  e `python.yml` (ruff lint + format + pytest).
- **ruff.toml** com regras pycodestyle/pyflakes/isort/bugbear.
- **CONTRIBUTING.md** com arquitetura, padrões de código e fluxo de trabalho.

### Atalhos de teclado
- `Alt+1` — Painel
- `Alt+2` — Documentos
- `Alt+3` — Clientes
- `Ctrl+T` — Alternar tema claro/escuro

### Removido
- `OContabil/Services/DominioExportService.cs` legado (substituído pelo novo em
  `OContabil/Services/Exports/`).
- `CRASH_OCONTABIL.txt` no Desktop substituído por logs estruturados em
  `%LocalAppData%\OContabil\logs\`.

### Notas de segurança
- O usuário padrão `admin/admin` é desabilitado pelo FirstRunWizard.
- O `Jwt:Key` em `appsettings.json` é um placeholder e **deve** ser substituído
  via variável de ambiente em produção (`JWT_KEY` no docker-compose).
