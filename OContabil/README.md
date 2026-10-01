# OContabil

[![.NET CI](https://github.com/vitorGgC569/OContabil/actions/workflows/dotnet.yml/badge.svg)](https://github.com/vitorGgC569/OContabil/actions/workflows/dotnet.yml)
[![Python CI](https://github.com/vitorGgC569/OContabil/actions/workflows/python.yml/badge.svg)](https://github.com/vitorGgC569/OContabil/actions/workflows/python.yml)
[![Licença](https://img.shields.io/badge/licença-MIT-blue.svg)](LICENSE)

**OContabil** é uma plataforma de processamento inteligente de documentos
contábeis com IA local, projetada para escritórios de contabilidade brasileiros.
Combina um aplicativo desktop WPF para uso diário com uma API REST para
integração SaaS, ambos compartilhando o mesmo motor de extração baseado em
**GLiNER 2** (ONNX nativo + fallback Python + regex offline).

## Sumário

- [Por que OContabil](#por-que-ocontabil)
- [Capacidades](#capacidades)
- [Arquitetura](#arquitetura)
- [Instalação rápida](#instalação-rápida)
- [Uso da API REST](#uso-da-api-rest)
- [Docker](#docker)
- [Roadmap](#roadmap)
- [Como contribuir](#como-contribuir)

## Por que OContabil

| Problema do contador                       | Solução OContabil                                    |
|--------------------------------------------|-----------------------------------------------------|
| Digitação manual de centenas de notas/mês  | Extração automática de NF-e, DANFE, boleto, holerite|
| Lançamentos divergentes em Domínio/SPED    | Exportação nativa em 5 formatos do mercado          |
| Dados sensíveis enviados para nuvem        | Processamento 100% local — nada sai da máquina       |
| Sem trilha de auditoria                    | AuditLogs em SQLite com diff de cada revisão        |
| Sem retomada após queda                    | Fila persistida + backup automático rotativo        |

## Capacidades

### Tipos de documento suportados
- **NF-e**, **NFS-e**, **CT-e**, **DANFE** — chave, número, valor, datas
- **Boleto bancário** — linha digitável, beneficiário, vencimento, valor
- **DARF** — código de receita, período, principal/multa/juros
- **Extrato bancário OFX** — transações com data, tipo, valor, memo
- **Holerite** — funcionário, líquido, descontos, encargos
- **Comprovante PIX** — valor, beneficiário, end-to-end ID
- **Schemas customizados** por cliente e tipo

### Exportações
- **Excel** (xlsx, 3 abas) — ClosedXML
- **CSV** UTF-8 BOM para Excel pt-BR
- **SPED Fiscal** — registros 0000/0001/0150/C100/9999 (ATO COTEPE 44/2018)
- **Domínio Sistemas** — layout pipe-delimitado de lançamentos contábeis
- **PDF de Conferência** — QuestPDF com cabeçalho da empresa

### Segurança
- Hash de senhas **PBKDF2-SHA256** com 120.000 iterações e salt
- Rate limiting de login (5 tentativas / 10 min → bloqueio 15 min)
- Timeout de sessão por inatividade
- **AuditLogs** detalhado em SQLite
- Backup automático rotativo do banco

### Operação
- Dashboard com gráficos LiveChartsCore (status, top clientes, confiança 14 dias)
- Banner de alerta de documentos parados acima do limite configurado
- Revisão com 3 abas: original / campos editáveis / histórico de revisões
- Motivo de rejeição obrigatório
- Atalhos de teclado (Alt+1/2/3, Ctrl+T)

## Arquitetura

```
┌────────────────────────────────┐    ┌──────────────────────────┐
│   Desktop WPF (.NET 8)         │    │  Frontend SaaS (HTML/JS)  │
│   ─ OContabil.exe              │    │  ─ saas/static/           │
└─────────────┬──────────────────┘    └──────────────┬───────────┘
              │                                       │
              │  in-process / SQLite local            │  HTTP+JWT
              │                                       │
              ▼                                       ▼
┌────────────────────────────────┐    ┌──────────────────────────┐
│   Services (compartilhados)    │    │  OContabil.API (ASP.NET) │
│   ─ PasswordHasher PBKDF2      │    │  ─ /api/auth (JWT)        │
│   ─ DocumentProcessingQueue    │    │  ─ /api/clients           │
│   ─ ExportManager (5 formatos) │    │  ─ /api/documents/upload  │
│   ─ AuditLogger                │    │  ─ /swagger               │
└─────────────┬──────────────────┘    └──────────────┬───────────┘
              │                                       │
              └────────────────┬──────────────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ Cadeia de motores de IA  │
                  │ ONNX → Python → Regex    │
                  └──────────────────────────┘
```

## Instalação rápida

### Desktop (Windows 10/11)
```bash
git clone https://github.com/vitorGgC569/OContabil.git
cd OContabil
dotnet run --project OContabil/OContabil.csproj
```
Primeiro login: `admin` / `admin` — o assistente força criação de novo admin
e desabilita o usuário padrão.

### Via instalador
Compile o `.exe` com Inno Setup 6 (veja [`installer/README.md`](installer/README.md)).

## Uso da API REST

```bash
# Login
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}'
# → { "token": "eyJhbGc...", "expiresIn": 28800 }

# Listar clientes
curl http://localhost:8080/api/clients -H "Authorization: Bearer $TOKEN"

# Upload de documento
curl -X POST http://localhost:8080/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@nota.pdf" -F "clientId=1" -F "documentType=NF-e"
```
Swagger interativo: http://localhost:8080/swagger

## Docker

Stack completa (API + GLiNER + Nginx):
```bash
export JWT_KEY="chave-de-32-caracteres-no-minimo-aqui!"
docker compose up --build
```
- API: http://localhost:8080
- GLiNER FastAPI: http://localhost:8000
- Nginx: http://localhost/

## Atalhos de teclado

| Atalho      | Ação                       |
|-------------|----------------------------|
| `Alt+1`     | Painel                     |
| `Alt+2`     | Documentos                 |
| `Alt+3`     | Clientes                   |
| `Ctrl+T`    | Alternar tema claro/escuro |
| `F5`        | Atualizar lista atual      |
| `Enter`     | Confirmar diálogo          |
| `Esc`       | Fechar diálogo             |

## Roadmap

Veja [`CHANGELOG.md`](CHANGELOG.md) para o histórico completo. Próximos passos:
- Multi-tenant na API (atualmente single-tenant local)
- Reconhecimento OCR direto via ONNX (eliminando Tesseract)
- WebSocket para acompanhar processamento em tempo real
- Integração SEFAZ para download direto de NF-e via certificado A1

## Como contribuir

Veja [`CONTRIBUTING.md`](CONTRIBUTING.md) para padrões de código, fluxo de PR
e como rodar os testes.

---

**Licença**: MIT — veja [`LICENSE`](LICENSE)
**Autor**: Vitor Gabriel Gomes — IF Goiano *campus* Urutaí, Sistemas de Informação
