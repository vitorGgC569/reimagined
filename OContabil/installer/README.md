# Empacotamento OContabil

Este diretório contém o script Inno Setup 6 (`OContabil.iss`) e instruções de
empacotamento do instalador `.exe`.

## Pré-requisitos

- **.NET 8 SDK** (para publicar)
- **Inno Setup 6** instalado (https://jrsoftware.org/isdl.php)
- (Opcional) **Embedded Python 3.11** baixado em `installer/python/`
- (Opcional) **Modelo GLiNER ONNX** baixado em `installer/models/onnx/`

## Passo a passo

1. **Publicar o aplicativo .NET self-contained**:
   ```cmd
   dotnet publish ..\OContabil\OContabil.csproj ^
     -c Release -r win-x64 --self-contained true ^
     /p:PublishSingleFile=false /p:PublishReadyToRun=true
   ```

2. **(Opcional) Baixar embedded Python**:
   ```cmd
   curl -L https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip ^
        -o python-embed.zip
   tar -xf python-embed.zip -C python\
   ```

3. **(Opcional) Baixar modelo GLiNER ONNX**:
   Coloque os arquivos `model.onnx` + `vocab.txt` em `models\onnx\`.

4. **Compilar o instalador**:
   ```cmd
   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" OContabil.iss
   ```
   O arquivo `Output\OContabil-Setup-1.0.0.exe` será gerado.

## Estrutura

```
installer/
├── OContabil.iss              # Script Inno Setup
├── README.md                  # Este arquivo
├── python/                    # (opcional) embedded Python
├── models/onnx/               # (opcional) modelo GLiNER ONNX
└── Output/                    # Onde o instalador final é gerado
```

## Tamanho aproximado

| Conteúdo                         | Tamanho |
|----------------------------------|---------|
| OContabil .NET self-contained    | ~80 MB  |
| Embedded Python 3.11             | ~10 MB  |
| Modelo GLiNER base (FP16)        | ~205 MB |
| Modelo GLiNER large (FP16)       | ~340 MB |
| **Total (instalação completa)**  | ~440 MB |
