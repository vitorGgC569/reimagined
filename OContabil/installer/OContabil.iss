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
; Binários principais (.NET 8 publicado em self-contained)
Source: "..\OContabil\bin\Release\net8.0-windows\publish\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Modelo GLiNER (opcional — só copia se existir na pasta)
Source: "models\onnx\*"; DestDir: "{app}\models\onnx"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Embedded Python (opcional — só copia se existir)
Source: "python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; Scripts Python
Source: "..\OContabil\Scripts\*"; DestDir: "{app}\Scripts"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  NetCoreInstalled: Boolean;
  ResultCode: Integer;
begin
  // Verifica .NET 8 Desktop Runtime via comando dotnet
  NetCoreInstalled := Exec('dotnet', '--list-runtimes', '', SW_HIDE,
                            ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not NetCoreInstalled then
  begin
    if MsgBox('O .NET 8 Desktop Runtime não foi detectado.' #13#13
              'O instalador continuará, mas será necessário instalar manualmente:' #13
              'https://dotnet.microsoft.com/download/dotnet/8.0' #13#13
              'Deseja continuar mesmo assim?',
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
      Exit;
    end;
  end;
  Result := True;
end;
