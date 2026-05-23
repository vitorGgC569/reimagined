# Contribuindo com o OContabil

Obrigado pelo interesse em contribuir! Este documento descreve o fluxo de trabalho,
padrões de código e como rodar o projeto localmente.

## Visão geral da arquitetura

```
OContabil/
├── OContabil/              # Desktop WPF (.NET 8 — Windows)
├── OContabil.API/          # API REST ASP.NET Core 8 (Linux/Windows/Docker)
├── OContabil.Tests/        # Testes xUnit + Moq + FluentAssertions
├── gliner2-service/        # Microsserviço FastAPI com GLiNER2 (Docker)
├── gliner2/                # Lib Python — modelo de extração
├── tests/                  # Testes Python (pytest)
├── saas/                   # Frontend HTML/JS conectado à API
├── installer/              # Script Inno Setup
└── docker-compose.yml      # Stack completa: API + GLiNER + Nginx
```

## Pré-requisitos

| Componente             | Versão | Onde                                  |
|------------------------|--------|---------------------------------------|
| .NET SDK               | 8.0    | https://dotnet.microsoft.com/         |
| Python                 | 3.11+  | https://python.org                    |
| Docker (opcional)      | 24+    | https://docker.com                    |
| Inno Setup (instalador)| 6+     | https://jrsoftware.org/isdl.php       |

## Rodando localmente

### Desktop
```bash
dotnet run --project OContabil/OContabil.csproj
```
Primeiro login: `admin` / `admin` — o assistente forçará a criação de um novo
admin. A senha padrão é desabilitada imediatamente.

### Testes
```bash
dotnet test OContabil.sln
pytest tests/ -v
```

### API + GLiNER + Nginx (Docker)
```bash
export JWT_KEY="sua-chave-de-32-caracteres-no-minimo!"
docker compose up --build
```
- API: http://localhost:8080/swagger
- GLiNER: http://localhost:8000/health
- Nginx: http://localhost/

### Instalador Windows
Veja `installer/README.md`.

## Padrões de código

### C# (.NET)
- Nullable habilitado (`<Nullable>enable</Nullable>`)
- File-scoped namespaces
- Records para DTOs
- Documentação `<summary>` em classes públicas com lógica não-trivial
- Sem warnings tratados como erros (build limpo é meta)

### Python
- Ruff para lint + format (`ruff check . && ruff format .`)
- Type hints obrigatórios em código novo
- Docstrings estilo Google
- `from __future__ import annotations` no topo

### Commits
Formato sugerido (não obrigatório):
```
<tipo>: <descrição curta>

<corpo opcional explicando o porquê>
```

Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

### Pull Requests
1. Crie uma branch a partir de `main`: `feat/minha-feature`
2. Faça commits pequenos e descritivos
3. Garanta que `dotnet test` e `pytest` passem
4. Abra o PR descrevendo: o que muda, por quê e como testar

## Estrutura de testes

| Camada            | Tecnologia        | Cobertura mínima |
|-------------------|-------------------|------------------|
| Services C#       | xUnit + Moq       | 60%              |
| Validators        | xUnit + Theory    | 100% dos paths   |
| Python extração   | pytest + fixtures | regressão        |
| API endpoints     | xUnit integration | golden path      |

## Segurança

- **Nunca** commite chaves JWT, senhas ou tokens
- Use `dotnet user-secrets` para credenciais em dev
- O `Jwt:Key` padrão em `appsettings.json` deve ser substituído em produção
- Rode `gh secret scanning` antes de abrir PRs sensíveis

## Reportando bugs

Abra uma issue com:
1. Versão do OContabil
2. Sistema operacional
3. Passos para reproduzir
4. Comportamento esperado vs. observado
5. Log de `%LocalAppData%\OContabil\logs\` (anonimize dados se necessário)

## Onde pedir ajuda

- Issues no GitHub
- Discussions (para perguntas abertas)
