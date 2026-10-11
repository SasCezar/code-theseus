# collect.ps1 — Run Arcan + collect architecture/context data for a single repo.
# Windows PowerShell / PowerShell Core (pwsh).
#
# Usage:
#   .\collect.ps1 -Repo C:\projects\myapp -Language JAVA -Output C:\data\myapp.jsonl
#
# Full options: Get-Help .\collect.ps1 -Detailed

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, HelpMessage="Path to the repository to analyse")]
    [string]$Repo,

    [Parameter(Mandatory=$true, HelpMessage="Language: C | CPP | JAVA | PYTHON | CSHARP")]
    [ValidateSet("C","CPP","JAVA","PYTHON","CSHARP", IgnoreCase=$true)]
    [string]$Language,

    [Parameter(Mandatory=$true, HelpMessage="Output JSONL file path")]
    [string]$Output,

    [Parameter(HelpMessage="Directory to store Arcan's raw output (default: <output-dir>\arcan_out)")]
    [string]$ArcanOutput = "",

    [Parameter(HelpMessage="Keep the Arcan output directory after collection")]
    [switch]$KeepArcanOutput,

    [Parameter(HelpMessage="Comma-separated context categories to omit: readme,docker,build,dependencies")]
    [string]$ExcludeCategories = "",

    [Parameter(HelpMessage="Max bytes per context file (default: 300000)")]
    [int]$MaxBytes = 300000,

    [int]$PhaseADepth = 2,
    [int]$PhaseAMax   = 20000,
    [int]$PhaseBDepth = 4,
    [int]$PhaseBMax   = 60000
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
$RepoPath   = (Resolve-Path $Repo).Path
$RepoName   = Split-Path $RepoPath -Leaf
$RepoParent = Split-Path $RepoPath -Parent

$OutputFile = [System.IO.Path]::GetFullPath($Output)
$OutputDir  = Split-Path $OutputFile -Parent
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if ($ArcanOutput -eq "") {
    $ArcanOutput = $OutputDir
}
$ArcanOutput = [System.IO.Path]::GetFullPath($ArcanOutput)
New-Item -ItemType Directory -Force -Path $ArcanOutput | Out-Null

# ── Check Docker ──────────────────────────────────────────────────────────────
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH.`nInstall Docker Desktop from https://docs.docker.com/get-docker/"
    exit 1
}

$ArcanImage = "ghcr.io/arcan-tech/arcan-2-cli-trial:latest"

# ── Step 1: Run Arcan ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Step 1/2: Running Arcan analysis ===================================="
Write-Host "  Repo:     $RepoPath"
Write-Host "  Language: $Language"
Write-Host "  Output:   $ArcanOutput\$RepoName"
Write-Host ""

# Mount parent dir so Arcan names output folder after the repo folder name
$RepoParentDocker = $RepoParent -replace "\\","/"
$ArcanOutputDocker = $ArcanOutput -replace "\\","/"

# Windows paths need to be converted to Docker Desktop format (e.g. C:\foo -> /c/foo)
function ConvertTo-DockerPath([string]$winPath) {
    $abs = [System.IO.Path]::GetFullPath($winPath)
    if ($abs -match "^([A-Za-z]):\\(.*)$") {
        $drive  = $Matches[1].ToLower()
        $rest   = $Matches[2] -replace "\\","/"
        return "/$drive/$rest"
    }
    return $abs -replace "\\","/"
}

$DockerRepoParent  = ConvertTo-DockerPath $RepoParent
$DockerArcanOutput = ConvertTo-DockerPath $ArcanOutput

docker run --rm --network none `
    -v "${DockerRepoParent}:/data/repos:ro" `
    -v "${DockerArcanOutput}:/data/arcan_out" `
    $ArcanImage `
    analyze `
    -i "/data/repos/$RepoName" `
    -o /data/arcan_out `
    -l $Language.ToUpper() `
    --all `
    output.writeDependencyGraph=true `
    output.writeMembership=true `
    output.writeDependency=true

if ($LASTEXITCODE -ne 0) {
    Write-Error "Arcan exited with code $LASTEXITCODE. See output above."
    exit $LASTEXITCODE
}

$ArcanRepoOutput = Join-Path $ArcanOutput "arcanOutput" $RepoName
if (-not (Test-Path $ArcanRepoOutput)) {
    Write-Error "Arcan did not produce output at $ArcanRepoOutput.`nCheck the Docker output above for errors."
    exit 1
}
Write-Host ""
Write-Host "  Arcan finished. Output at: $ArcanRepoOutput"

# ── Step 2: Collect data ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Step 2/2: Collecting data ============================================"

# Find Python
$PythonCmd = $null
foreach ($candidate in @("python3","python","py")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $PythonCmd = $candidate
        break
    }
}
if (-not $PythonCmd) {
    Write-Error "Python 3 is not installed or not in PATH.`nDownload from https://www.python.org/downloads/"
    exit 1
}

$PyVersion = & $PythonCmd --version 2>&1
Write-Host "  Using: $PyVersion"

$ExtraArgs = @()
if ($ExcludeCategories -ne "") {
    $ExtraArgs += "--exclude-categories"
    $ExtraArgs += $ExcludeCategories
}

& $PythonCmd "$ScriptDir\collect_repo.py" `
    --repo-path $RepoPath `
    --arcan-output $ArcanRepoOutput `
    --language $Language `
    --output $OutputFile `
    --max-bytes $MaxBytes `
    --phaseA-depth $PhaseADepth `
    --phaseA-max-files $PhaseAMax `
    --phaseB-depth $PhaseBDepth `
    --phaseB-max-files $PhaseBMax `
    @ExtraArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "collect_repo.py exited with code $LASTEXITCODE."
    exit $LASTEXITCODE
}

# ── Cleanup ───────────────────────────────────────────────────────────────────
if (-not $KeepArcanOutput) {
    Write-Host ""
    Write-Host "  Removing Arcan output directory ($ArcanRepoOutput)..."
    Remove-Item -Recurse -Force $ArcanRepoOutput
}

Write-Host ""
Write-Host "========================================================================="
Write-Host "  Done!  Output: $OutputFile"
Write-Host "========================================================================="
