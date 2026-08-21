[CmdletBinding()]
param(
    [string]$NuGetPath
)

$ErrorActionPreference = "Stop"
$crate = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $crate
$stage = Join-Path $PSScriptRoot "stage"
$output = Join-Path $repo "target\nuget"

if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage, $output | Out-Null

Push-Location $repo
try {
    cargo build -p diskann-memory-ffi
    if ($LASTEXITCODE -ne 0) { throw "Debug build failed" }
    cargo build -p diskann-memory-ffi --release
    if ($LASTEXITCODE -ne 0) { throw "Release build failed" }
} finally {
    Pop-Location
}

$copies = @{
    (Join-Path $crate "include\diskann_memory_ffi.h") = "include\diskann_memory_ffi.h"
    (Join-Path $crate "docs\README.md") = "docs\README.md"
    (Join-Path $PSScriptRoot "RustDiskANNFFI.Library.targets") = "build\RustDiskANNFFI.Library.targets"
    (Join-Path $repo "target\debug\diskann_memory_ffi.dll") = "x64\bin\debug\diskann_memory_ffi.dll"
    (Join-Path $repo "target\debug\diskann_memory_ffi.pdb") = "x64\bin\debug\diskann_memory_ffi.pdb"
    (Join-Path $repo "target\debug\diskann_memory_ffi.dll.lib") = "x64\lib\debug\diskann_memory_ffi.lib"
    (Join-Path $repo "target\release\diskann_memory_ffi.dll") = "x64\bin\release\diskann_memory_ffi.dll"
    (Join-Path $repo "target\release\diskann_memory_ffi.pdb") = "x64\bin\release\diskann_memory_ffi.pdb"
    (Join-Path $repo "target\release\diskann_memory_ffi.dll.lib") = "x64\lib\release\diskann_memory_ffi.lib"
}
foreach ($entry in $copies.GetEnumerator()) {
    $destination = Join-Path $stage $entry.Value
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $entry.Key -Destination $destination -Force
}

& (Join-Path $PSScriptRoot "validate-consumer.ps1") -StageRoot $stage

$nuget = $null
if ($NuGetPath) {
    $nuget = Get-Item -LiteralPath $NuGetPath
} else {
    $nuget = Get-Command nuget.exe -ErrorAction SilentlyContinue
    if (-not $nuget) {
        $nuget = Get-Command nuget -ErrorAction SilentlyContinue
    }
}
if (-not $nuget) {
    throw "nuget.exe was not found; add it to PATH or pass -NuGetPath"
}
$nugetCommand = if ($nuget.PSPath) { $nuget.FullName } else { $nuget.Source }
$package = Join-Path $output "RustDiskANNFFI.Library.0.2.0.nupkg"
if (Test-Path -LiteralPath $package) {
    Remove-Item -LiteralPath $package -Force
}
& $nugetCommand pack (Join-Path $PSScriptRoot "RustDiskANNFFI.Library.nuspec") -OutputDirectory $output -NoPackageAnalysis -NonInteractive
if ($LASTEXITCODE -ne 0) {
    throw "NuGet pack failed"
}

$hash = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "$package"
Write-Host "SHA256 $hash"
