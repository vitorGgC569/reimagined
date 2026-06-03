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
- (Opcional) **Embedded Python 3.11** em `installer/python/` — habilita o motor GLiNER2
  sem o usuário instalar Python
- (Opcional) **Modelo GLiNER ONNX** em `installer/models/onnx/`

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

3. **(Opcional) Embedded Python / modelo GLiNER** em `python/` e `models/onnx/`.

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
├── models/onnx/           # (opcional) modelo GLiNER ONNX
└── Output/                # (gerado) instalador final .exe
```

## Tamanho aproximado

| Conteúdo                              | Tamanho  |
|---------------------------------------|----------|
| OContabil self-contained (verificado) | ~345 MB  |
| Embedded Python 3.11 (opcional)       | ~10 MB   |
| Modelo GLiNER base (opcional)         | ~205 MB  |

> O motor de extração padrão (determinístico + OCR + NF-e XML estruturada) funciona
> 100% offline sem Python nem modelo ONNX; GLiNER2 é um upgrade opcional.
