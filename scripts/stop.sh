#!/usr/bin/env bash
# End of session: append a short entry to docs/session-log.md capturing what
# changed, then stop the dependencies.
#
# Usage:  bash scripts/stop.sh [--full] [--note "..."]
#
#   --full   also quit the Ollama app and Docker Desktop entirely
#            (default: stop the Qdrant container + unload Ollama models)
#   --note   optional one-line summary, added to the log
#
# Git Bash equivalent of scripts/stop.ps1 (shares docs/session-log.md).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

LOG="docs/session-log.md"
QDRANT="http://localhost:6333"
COLLECTION="insurance_ca_v1"
MODELS=("llama3.2:3b" "nomic-embed-text")
STAMP="$(date '+%Y-%m-%d %H:%M')"
FULL=0
NOTE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --note) NOTE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

info() { printf '  %s\n' "$1"; }

# --- API/UI auto-stop: built, tested, and working -- disabled by request. ---
# Kept here (commented) rather than deleted in case it's wanted again later.
# Kills any process whose command line contains BOTH substrings (never just
# a port or a process name like "python.exe" -- too broad, could hit VS
# Code's language server, Jupyter, anything). Requiring two specific
# substrings together is precise enough that nothing else would ever match.
# Safe to call every time: finds nothing and no-ops if already stopped.
#
# stop_by_cmdline() {
#   local pat1="$1" pat2="$2" label="$3"
#   local pids
#   # NOTE: this powershell.exe invocation's own command line necessarily
#   # contains "$pat1"/"$pat2" (it has to, to search for them) -- Get-CimInstance
#   # enumerates ALL processes including itself, so without excluding $PID
#   # (its own automatic variable) it always self-matches, reporting a false
#   # "stopped" every run even with nothing actually running. Caught by
#   # actually re-running this script three times and noticing it never
#   # settled to "not running" -- verify, don't assume.
#   pids="$(powershell.exe -NoProfile -Command \
#     "(Get-CimInstance Win32_Process | Where-Object { \$_.ProcessId -ne \$PID -and \$_.CommandLine -like '*${pat1}*' -and \$_.CommandLine -like '*${pat2}*' }).ProcessId" \
#     2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$')"
#   if [ -n "$pids" ]; then
#     for pid in $pids; do
#       powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
#     done
#     info "stopped $label (pid: $(echo "$pids" | tr '\n' ' '))"
#   else
#     info "$label not running"
#   fi
# }

echo
echo "=== insurance-rag :: stop ==="

# 1. Gather state ---------------------------------------------------------
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(not a git repo)')"
head="$(git log -1 --pretty='%h %s' 2>/dev/null || echo '(no commits yet)')"
mapfile -t changed < <(git status --porcelain -uall 2>/dev/null | sed '/^[[:space:]]*$/d')

prev="$(grep -oE 'HEAD `[0-9a-f]{7,40}' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9a-f]{7,40}')"
if [ -n "$prev" ]; then
  mapfile -t commits < <(git log "$prev..HEAD" --pretty='%h %s' 2>/dev/null)
else
  mapfile -t commits < <(git log -10 --pretty='%h %s' 2>/dev/null)
fi

pts="$(curl -sf -m 3 "$QDRANT/collections/$COLLECTION" 2>/dev/null \
       | grep -oE '"points_count": *[0-9]+' | grep -oE '[0-9]+')"
[ -n "$pts" ] || pts="(qdrant not reachable)"

# 2. Append log entry --------------------------------------------------
mkdir -p "$(dirname "$LOG")"
{
  if [ ! -f "$LOG" ]; then
    printf '# Session Log\n\nAppended by `scripts/stop.ps1` / `scripts/stop.sh` at the end of each working session.\n'
  fi
  printf '\n## %s\n' "$STAMP"
  [ -n "$NOTE" ] && printf -- '- **Note:** %s\n' "$NOTE"
  printf -- '- Branch `%s`, HEAD `%s`\n' "$branch" "$head"
  printf -- '- Qdrant `%s`: %s points\n' "$COLLECTION" "$pts"
  if [ "${#commits[@]}" -gt 0 ] && [ -n "${commits[0]}" ]; then
    printf -- '- Commits this session (%s):\n' "${#commits[@]}"
    for c in "${commits[@]}"; do printf -- '  - `%s`\n' "$c"; done
  else
    printf -- '- Commits this session: none\n'
  fi
  if [ "${#changed[@]}" -gt 0 ]; then
    printf -- '- Uncommitted (%s):\n' "${#changed[@]}"
    for l in "${changed[@]}"; do printf -- '  - `%s`\n' "$(printf '%s' "$l" | sed 's/^[[:space:]]*//')"; done
  else
    printf -- '- Uncommitted: clean\n'
  fi
} >> "$LOG"
info "Logged to $LOG  (${#commits[@]} commit(s), ${#changed[@]} uncommitted)"

# 3. Stop dependencies -----------------------------------------------
# stop_by_cmdline "uvicorn" "src.api.main" "API (uvicorn)"
# stop_by_cmdline "streamlit" "run ui" "UI (streamlit)"
info "Reminder: close the API/UI dev servers yourself (Ctrl+C in their terminals) --"
info "this script only stops Qdrant/Ollama, not uvicorn/streamlit."

info "Stopping Qdrant container..."
docker compose stop >/dev/null 2>&1

info "Unloading Ollama models..."
for m in "${MODELS[@]}"; do ollama stop "$m" >/dev/null 2>&1; done

if [ "$FULL" = 1 ]; then
  info "Quitting Ollama + Docker Desktop..."
  powershell.exe -NoProfile -Command \
    "Get-Process 'ollama app','ollama','Docker Desktop' -ErrorAction SilentlyContinue | Stop-Process -Force" \
    2>/dev/null
fi

echo
echo "Done. Safe to close."
echo
