; OContabil — Script de instalação Inno Setup 6
; Gera um instalador único .exe com binário .NET, modelo opcional GLiNER e
; opção de embedded Python para usuários sem Python instalado.

#define MyAppName "OContabil"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "OContabil"
#define MyAppURL "https://github.com/vitorGgC569/OContabil"
#define MyAppExeName "OContabil.exe"

[Setup]
AppId={{8F1D5C32-1A1B-4E4E-8F71-001A1B4E4E00}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=OContabil-Setup-{#MyAppVersion}
SetupIconFile=..\OContabil\Assets\logo.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}

[Languages]
Name: "brazilian"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Binários principais (.NET 8 publicado em self-contained — gerado por build_installer.ps1 em .\publish)
Source: "publish\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Modelo GLiNER ONNX (opcional — caminho C# nativo, hoje dormente)
Source: "models\onnx\*"; DestDir: "{app}\models\onnx"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Modelo GLiNER2 (gliner2-multi-v1) — cache HuggingFace embutido p/ uso 100% OFFLINE (opcional, ~ver README).
; Quando presente, o app seta HF_HOME/HF_HUB_OFFLINE e NÃO baixa nada da internet.
Source: "models\hf\*"; DestDir: "{app}\models\hf"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Python embarcado + gliner2 (opcional — habilita o motor GLiNER sem Python instalado).
; O app detecta {app}\python\python.exe automaticamente. NOTA: inclui PyTorch (pesado).
Source: "python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Scripts Python
Source: "..\OContabil\Scripts\*"; DestDir: "{app}\Scripts"; Flags: ignoreversion recursesubdirs

; Bootstrapper do WebView2 Runtime (Evergreen) — baixado por build_installer.ps1; instalado só se ausente
Source: "redist\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist; Check: not WebView2Installed

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Instala o WebView2 Runtime (Evergreen) silenciosamente, se ausente e o bootstrapper foi empacotado.
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Instalando o Microsoft Edge WebView2 Runtime..."; Flags: skipifdoesntexist; Check: not WebView2Installed
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
{ App é self-contained (.NET 8 embarcado) — a única dependência de sistema é o WebView2 Runtime. }
{ Detecta o WebView2 (Evergreen) pelo GUID estável de cliente do EdgeUpdate. }
function WebView2Installed(): Boolean;
var
  pv: String;
begin
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', pv) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', pv) or
    RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', pv);
  if Result then
    Result := (pv <> '') and (pv <> '0.0.0.0');
end;

function InitializeSetup(): Boolean;
begin
  if (not WebView2Installed()) and (not FileExists(ExpandConstant('{src}\redist\MicrosoftEdgeWebview2Setup.exe'))) then
    MsgBox('O Microsoft Edge WebView2 Runtime não foi detectado e o instalador offline dele não está incluído.' #13#13
           'O OContabil precisa do WebView2 para a interface. Instale-o (gratuito):' #13
           'https://developer.microsoft.com/microsoft-edge/webview2/',
           mbInformation, MB_OK);
  Result := True;
end;
