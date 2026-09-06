<#
.SYNOPSIS
  End of session: append a short entry to docs/session-log.md capturing what
  changed, then stop the dependencies.

.USAGE
  powershell -ExecutionPolicy Bypass -File scripts\stop.ps1 [-Full] [-Note "..."]

  -Full   also quit the Ollama app and Docker Desktop entirely
          (default: stop the Qdrant container + unload Ollama models,
           leave the apps running so the next start is instant)
  -Note   optional one-line summary of the session, added to the log
#>
[CmdletBinding()]
param(
  [switch]$Full,
  [string]$Note
)

$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Info($m) { Write-Host "  $m" -ForegroundColor Cyan }

$LogFile    = Join-Path $RepoRoot 'docs\session-log.md'
$Qdrant     = 'http://localhost:6333'
$Collection = 'insurance_ca_v1'
$Models     = @('llama3.2:3b', 'nomic-embed-text')
$stamp      = Get-Date -Format 'yyyy-MM-dd HH:mm'

Write-Host "`n=== insurance-rag :: stop ===" -ForegroundColor White

# 1. Gather state -----------------------------------------------------------
$branch = (git rev-parse --abbrev-ref HEAD 2>$null)
if (-not $branch) { $branch = '(not a git repo)' }
$head = (git log -1 --pretty='%h %s' 2>$null)
if (-not $head) { $head = '(no commits yet)' }

# uncommitted work (expand untracked dirs so individual files show)
$changed = @(git status --porcelain -uall 2>$null | Where-Object { $_.Trim() })

# commits made since the previous log entry (that entry records the HEAD hash)
$prevHash = $null
if (Test-Path $LogFile) {
  $m = Select-String -Path $LogFile -Pattern 'HEAD ``([0-9a-f]{7,40})' |
       Select-Object -Last 1
  if ($m) { $prevHash = $m.Matches[0].Groups[1].Value }
}
if ($prevHash) { $sessionCommits = @(git log "$prevHash..HEAD" --pretty='%h %s' 2>$null) }
else           { $sessionCommits = @(git log -10 --pretty='%h %s' 2>$null) }

$points = '(qdrant not reachable)'
try { $points = (Invoke-RestMethod "$Qdrant/collections/$Collection" -TimeoutSec 3).result.points_count } catch { }

# 2. Append log entry -----------------------------------------------------
$lines = @()
$lines += ""
$lines += "## $stamp"
if ($Note) { $lines += "- **Note:** $Note" }
$lines += "- Branch ``$branch``, HEAD ``$head``"
$lines += "- Qdrant ``$Collection``: $points points"
if ($sessionCommits.Count -gt 0) {
  $lines += "- Commits this session ($($sessionCommits.Count)):"
  foreach ($c in $sessionCommits) { $lines += "  - ``$c``" }
} else {
  $lines += "- Commits this session: none"
}
if ($changed.Count -gt 0) {
  $lines += "- Uncommitted ($($changed.Count)):"
  foreach ($l in $changed) { $lines += "  - ``$($l.Trim())``" }
} else {
  $lines += "- Uncommitted: clean"
}

$logDir = Split-Path $LogFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
if (-not (Test-Path $LogFile)) {
  $lines = @("# Session Log", "",
             "Appended by ``scripts/stop.ps1`` at the end of each working session.") + $lines
}
$utf8 = New-Object System.Text.UTF8Encoding($false)   # no BOM
[System.IO.File]::AppendAllText($LogFile, (($lines -join "`n") + "`n"), $utf8)
Info "Logged to docs/session-log.md  ($($sessionCommits.Count) commit(s), $($changed.Count) uncommitted)"

# 3. Stop dependencies ----------------------------------------------------
Info "Stopping Qdrant container..."
docker compose stop 2>$null | Out-Null

Info "Unloading Ollama models..."
foreach ($m in $Models) { ollama stop $m 2>$null | Out-Null }

if ($Full) {
  Info "Quitting Ollama app..."
  Get-Process -Name 'ollama app', 'ollama' -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
  Info "Quitting Docker Desktop..."
  Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
}

Write-Host "`nDone. Safe to close." -ForegroundColor White
Write-Host ""
