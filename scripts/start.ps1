<#
.SYNOPSIS
  Bring up everything needed to work on insurance-rag:
  Docker Desktop -> Qdrant container -> Ollama + models, then sanity-check the
  vector collection and warm the models.

.USAGE
  powershell -ExecutionPolicy Bypass -File scripts\start.ps1 [-NoWarm]
#>
[CmdletBinding()]
param(
  [switch]$NoWarm   # skip the model warm-up calls
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Info($m) { Write-Host "  $m"      -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  OK  $m"  -ForegroundColor Green }
function Warn($m) { Write-Host "  !!  $m"  -ForegroundColor Yellow }

$Models     = @('llama3.2:3b', 'nomic-embed-text')
$Qdrant     = 'http://localhost:6333'
$Ollama     = 'http://localhost:11434'
$Collection = 'insurance_ca_v1'

Write-Host "`n=== insurance-rag :: start ===" -ForegroundColor White

# 1. Docker Desktop -----------------------------------------------------------
Info "Checking Docker daemon..."
$null = (docker info 2>$null)
if ($LASTEXITCODE -ne 0) {
  Warn "Docker not responding - launching Docker Desktop"
  $dd = Join-Path $Env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
  if (-not (Test-Path $dd)) { throw "Docker Desktop not found at $dd" }
  Start-Process $dd
  $deadline = (Get-Date).AddMinutes(3)
  do {
    Start-Sleep 4
    $null = (docker info 2>$null)
  } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
  if ($LASTEXITCODE -ne 0) { throw "Docker did not become ready within 3 minutes" }
}
Ok "Docker is up"

# 2. Qdrant -----------------------------------------------------------------
Info "Starting Qdrant (docker compose up -d)..."
docker compose up -d | Out-Null
if ($LASTEXITCODE -ne 0) {
  # a stray (non-compose) container may be holding port 6333
  $stray = @(docker ps --filter "publish=6333" --format "{{.Names}}" |
             Where-Object { $_ -and $_ -ne 'insurance-rag-qdrant' })
  if ($stray.Count -gt 0) {
    Warn "removing stray container(s) on port 6333: $($stray -join ', ')"
    foreach ($s in $stray) { docker rm -f $s | Out-Null }
    docker compose up -d | Out-Null
  }
  if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
}
$ready = $false
$deadline = (Get-Date).AddSeconds(60)
do {
  Start-Sleep 2
  try { Invoke-RestMethod "$Qdrant/collections" -TimeoutSec 3 | Out-Null; $ready = $true } catch { }
} while (-not $ready -and (Get-Date) -lt $deadline)
if (-not $ready) { throw "Qdrant not responding on $Qdrant" }
Ok "Qdrant ready on $Qdrant"

# 3. Ollama ---------------------------------------------------------------
Info "Checking Ollama..."
$olive = $false
try { Invoke-RestMethod "$Ollama/api/tags" -TimeoutSec 3 | Out-Null; $olive = $true } catch { }
if (-not $olive) {
  Warn "Ollama not responding - starting 'ollama serve'"
  Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden
  $deadline = (Get-Date).AddSeconds(30)
  do {
    Start-Sleep 2
    try { Invoke-RestMethod "$Ollama/api/tags" -TimeoutSec 3 | Out-Null; $olive = $true } catch { }
  } while (-not $olive -and (Get-Date) -lt $deadline)
  if (-not $olive) { throw "Ollama did not start" }
}
Ok "Ollama is up"

# 3b. Required models  (normalise ':latest' so names compare correctly)
function Norm($n) { if ($n -match ':') { $n } else { "${n}:latest" } }
$have = @(ollama list | Select-Object -Skip 1 |
          ForEach-Object { Norm (($_ -split '\s+')[0]) })
foreach ($m in $Models) {
  if ($have -contains (Norm $m)) { Ok "model $m present" }
  else { Warn "pulling $m (first time only)..."; ollama pull $m }
}

# 3c. Warm models so the first real query is fast (keep_alive matches settings)
if (-not $NoWarm) {
  Info "Warming models (keep_alive 30m)..."
  try {
    Invoke-RestMethod "$Ollama/api/generate" -Method Post -TimeoutSec 120 -ContentType 'application/json' `
      -Body (@{ model = 'llama3.2:3b'; prompt = 'ok'; stream = $false; keep_alive = '30m' } | ConvertTo-Json) | Out-Null
    Invoke-RestMethod "$Ollama/api/embed" -Method Post -TimeoutSec 60 -ContentType 'application/json' `
      -Body (@{ model = 'nomic-embed-text'; input = 'ok'; keep_alive = '30m' } | ConvertTo-Json) | Out-Null
    Ok "models warm"
  } catch { Warn "warm-up skipped ($($_.Exception.Message))" }
}

# 4. Vector collection ------------------------------------------------------
Info "Checking Qdrant collection '$Collection'..."
try {
  $c = Invoke-RestMethod "$Qdrant/collections/$Collection" -TimeoutSec 5
  Ok "collection '$Collection' exists ($($c.result.points_count) points)"
} catch {
  Warn "collection '$Collection' missing - run:  python -m src.ingestion.bootstrap_collection"
}

Write-Host ""
Write-Host "Ready. If your shell isn't in the venv yet:  venv\Scripts\activate" -ForegroundColor White
Write-Host ""
