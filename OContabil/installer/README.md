# Empacotamento OContabil

Gera um instalador Windows (`.exe`) **self-contained** — não exige .NET na máquina do
usuário. A única dependência de sistema é o **Microsoft Edge WebView2 Runtime** (padrão
no Windows 11; o instalador o instala automaticamente se o bootstrapper estiver em `redist/`).

## Forma rápida (recomendada)

```powershell
cd OContabil/installer
pwsh -File build_installer.ps1
```

O script faz tudo: `dotnet publish` self-contained em `publish/`, baixa o bootstrapper
do WebView2 em `redist/`, e compila `OContabil.iss` com o `iscc`. Saída em
`Output/OContabil-Setup-1.0.0.exe`.

## Pré-requisitos
- **.NET 8 SDK** (para publicar)
- **Inno Setup 6** (https://jrsoftware.org/isdl.php) — `iscc.exe` no PATH ou em
  `C:\Program Files (x86)\Inno Setup 6`
- (Opcional) **GLiNER 100% offline**: Python embarcado em `installer/python/` + cache do
  modelo `gliner2-multi-v1` em `installer/models/hf/` — ver seção "GLiNER 100% offline" abaixo

## Passo a passo manual

1. **Publicar self-contained** (a pasta `publish/` é o que o `.iss` empacota):
   ```powershell
   dotnet publish ..\OContabil\OContabil.csproj -c Release -r win-x64 `
     --self-contained true -o publish
   ```
   Inclui: `OContabil.exe` + runtime .NET 8 + DLLs nativas (SQLCipher `e_sqlcipher.dll`,
   Tesseract `x64\tesseract50.dll`+`leptonica`, `pdfium.dll`, `libSkiaSharp.dll`) +
   `tessdata/` (OCR por+eng) + `Scripts/` + `WebUI/`.

2. **(Opcional) Bootstrapper do WebView2** em `redist/` (o `build_installer.ps1` baixa
   sozinho; manualmente: salve o "Evergreen Bootstrapper" de
   https://developer.microsoft.com/microsoft-edge/webview2/ como
   `redist\MicrosoftEdgeWebview2Setup.exe`). Sem ele, o instalador apenas exibe instrução
   de instalação manual quando o WebView2 estiver ausente.

3. **(Opcional) GLiNER 100% offline** — ver seção dedicada abaixo (`python/` + `models/hf/`).

4. **Compilar o instalador**:
   ```powershell
   & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" OContabil.iss
   ```

## Estrutura
```
installer/
├── OContabil.iss          # Script Inno Setup (self-contained + WebView2)
├── build_installer.ps1    # Build de ponta a ponta (publish + redist + iscc)
├── README.md
├── publish/               # (gerado) saída do dotnet publish
├── redist/                # (gerado) MicrosoftEdgeWebview2Setup.exe
├── python/                # (opcional) embedded Python
├── models/onnx/           # (opcional) modelo GLiNER ONNX (caminho C# dormente)
├── models/hf/             # (opcional) cache HuggingFace do gliner2-multi-v1 (offline)
└── Output/                # (gerado) instalador final .exe
```

## GLiNER 100% offline (opcional) — gliner2-multi-v1

O motor padrão (determinístico + OCR + NF-e XML estruturada) já roda **100% offline**
sem Python nem modelo. O **GLiNER2** (modelo padrão: `fastino/gliner2-multi-v1`,
multilíngue) é um upgrade opcional para documentos não-estruturados; para rodá-lo
offline, embuta o Python + o modelo:

1. **Python embarcado + gliner2** em `installer/python/`:
   ```powershell
   curl -L https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip -o py.zip
   Expand-Archive py.zip -DestinationPath python
   # habilite o site no python311._pth (descomente "import site"), instale o pip e:
   .\python\python.exe -m pip install gliner2   # puxa torch/transformers (pesado)
   ```
   O app detecta `{app}\python\python.exe` automaticamente (sem Python no sistema).
   **OCR não é necessário no Python**: o C# já faz o OCR (Tesseract + PDFium) e envia o texto
   pronto ao sidecar — dispensa `pytesseract`/`Pillow`/`PyMuPDF` no bundle.

2. **Cache do modelo** em `installer/models/hf/` (baixa uma vez; depois fica offline):
   ```powershell
   $env:HF_HOME = "$PWD\models\hf"
   .\python\python.exe -c "from gliner2 import GLiNER2; GLiNER2.from_pretrained('fastino/gliner2-multi-v1')"
   ```
   Em runtime o app seta `HF_HOME={app}\models\hf` + `HF_HUB_OFFLINE=1` quando essa pasta
   existe — então **não baixa nada**.

## Tamanho aproximado

| Conteúdo                                    | Tamanho     |
|---------------------------------------------|-------------|
| Núcleo self-contained (verificado)          | ~345 MB     |
| + Python embarcado **com gliner2/PyTorch**  | ~2–2,5 GB   |
| + Modelo gliner2-multi-v1 (cache HF)        | ~0,9–1,5 GB |

> **Trade-off honesto:** o núcleo offline é leve (~345 MB) e cobre bem os casos comuns.
> Embutir o GLiNER offline incha o instalador para **vários GB** (por causa do PyTorch).
> Alternativas: deixar o app baixar o modelo na 1ª vez (exige internet uma vez) ou exigir
> Python+gliner2 instalado pelo usuário. O núcleo continua 100% offline em qualquer caso.
