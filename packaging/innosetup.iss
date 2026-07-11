; innosetup.iss — Inno Setup script for LT2 Thermometry v1.0 Windows installer.
;
; Prerequisites:
;   1. Run  packaging\build_pyinstaller.bat  so that
;      dist\LT2 Thermometry\  exists and contains the bundle.
;   2. Install Inno Setup 6.x from https://jrsoftware.org/isinfo.php
;   3. Open this file in the Inno Setup IDE and click Compile,
;      OR run from the command line:
;         "C:\Program Files (x86)\Inno Setup 6\iscc.exe" packaging\innosetup.iss
;
; Output: packaging\Output\LT2_Thermometry_1.0_win64_setup.exe

#define AppName      "SpectraSensML"
#define AppVersion   "1.0"
#define AppPublisher "Aleksandar Ciric, OMAS group"
#define AppURL       "https://www.omasgroup.org"
#define AppExeName   "SpectraSensML.exe"
; Adjust SourceDir to the repo root (Yb/) relative to this .iss file.
#define SourceDir    "..\dist\SpectraSensML"

[Setup]
AppId={{E9A2F7B3-1C4D-4E5F-8A2B-0D3C6E7F8901}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Output installer file
OutputDir=..\packaging\Output
OutputBaseFilename=SpectraSensML_{#AppVersion}_win64_setup
SetupIconFile=..\lt2_gui\assets\app_icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
; Require 64-bit Windows
ArchitecturesInstallIn64BitMode=x64os
; Request admin rights so it installs to Program Files
PrivilegesRequired=admin
; Ask the user if they want a desktop shortcut
InfoBeforeFile=
; Minimum OS version: Windows 10 (6.2 = Win 8, 10.0 = Win 10)
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle folder created by PyInstaller
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\{#AppName}";          Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
; Optional desktop shortcut (only if task selected)
Name: "{autodesktop}\{#AppName}";    Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Offer to launch after install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove the folder entirely on uninstall (PyInstaller onedir leaves no registry cruft)
Type: filesandordirs; Name: "{app}"

[Code]
// No custom code required for basic install/uninstall.
