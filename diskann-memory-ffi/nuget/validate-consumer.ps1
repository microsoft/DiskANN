[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StageRoot
)

$ErrorActionPreference = "Stop"
$StageRoot = (Resolve-Path -LiteralPath $StageRoot).Path
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe was not found"
}
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    throw "Visual C++ x64 tools were not found"
}
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$source = Join-Path $PSScriptRoot "consumer.cpp"
$outRoot = Join-Path $PSScriptRoot "consumer-out"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

foreach ($configuration in @("debug", "release")) {
    $out = Join-Path $outRoot $configuration
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Copy-Item -LiteralPath (Join-Path $StageRoot "x64\bin\$configuration\diskann_memory_ffi.dll") -Destination $out -Force
    $include = Join-Path $StageRoot "include"
    $lib = Join-Path $StageRoot "x64\lib\$configuration"
    $exe = Join-Path $out "consumer.exe"
    $command = "call `"$vcvars`" >nul && cl /nologo /std:c++17 /EHsc /I`"$include`" `"$source`" /link /LIBPATH:`"$lib`" diskann_memory_ffi.lib /OUT:`"$exe`" && `"$exe`""
    & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "$configuration C++ consumer validation failed"
    }
}
