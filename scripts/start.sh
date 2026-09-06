#!/usr/bin/env bash
# Bring up everything needed to work on insurance-rag:
# Docker Desktop -> Qdrant container -> Ollama + models, warm, sanity-check.
#
# Usage:  bash scripts/start.sh [--no-warm]
#
# Git Bash equivalent of scripts/start.ps1.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

QDRANT="http://localhost:6333"
OLLAMA="http://localhost:11434"
COLLECTION="insurance_ca_v1"
MODELS=("llama3.2:3b" "nomic-embed-text")
NO_WARM=0
[ "$1" = "--no-warm" ] && NO_WARM=1

info() { printf '  %s\n' "$1"; }
ok()   { printf '  OK  %s\n' "$1"; }
warn() { printf '  !!  %s\n' "$1"; }
die()  { printf '  XX  %s\n' "$1" >&2; exit 1; }

echo
echo "=== insurance-rag :: start ==="

# 1. Docker daemon ----------------------------------------------------------
info "Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
  warn "Docker not responding - launching Docker Desktop"
  DD="${PROGRAMFILES:-/c/Program Files}/Docker/Docker/Docker Desktop.exe"
  if [ -f "$DD" ]; then
    "$DD" &
  else
    powershell.exe -NoProfile -Command 'Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"' 2>/dev/null \
      || die "Docker Desktop not found"
  fi
  for _ in $(seq 1 45); do
    docker info >/dev/null 2>&1 && break
    sleep 4
  done
  docker info >/dev/null 2>&1 || die "Docker did not become ready within 3 minutes"
fi
ok "Docker is up"

# 2. Qdrant via compose ---------------------------------------------------
info "Starting Qdrant (docker compose up -d)..."
if ! docker compose up -d >/dev/null 2>&1; then
  stray="$(docker ps --filter 'publish=6333' --format '{{.Names}}' | grep -v '^insurance-rag-qdrant$' || true)"
  if [ -n "$stray" ]; then
    warn "removing stray container(s) on port 6333: $stray"
    echo "$stray" | xargs -r docker rm -f >/dev/null
    docker compose up -d >/dev/null 2>&1 || die "docker compose up failed"
  else
    die "docker compose up failed"
  fi
fi
ready=0
for _ in $(seq 1 30); do
  curl -sf -m 3 "$QDRANT/collections" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
done
[ "$ready" = 1 ] || die "Qdrant not responding on $QDRANT"
ok "Qdrant ready on $QDRANT"

# 3. Ollama -------------------------------------------------------------
info "Checking Ollama..."
if ! curl -sf -m 3 "$OLLAMA/api/tags" >/dev/null 2>&1; then
  warn "Ollama not responding - starting 'ollama serve'"
  nohup ollama serve >/dev/null 2>&1 &
  for _ in $(seq 1 15); do
    curl -sf -m 3 "$OLLAMA/api/tags" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf -m 3 "$OLLAMA/api/tags" >/dev/null 2>&1 || die "Ollama did not start"
fi
ok "Ollama is up"

# 3b. Required models (normalise ':latest' so names compare correctly)
norm() { case "$1" in *:*) echo "$1";; *) echo "$1:latest";; esac; }
have="$(ollama list | awk 'NR>1{print $1}' | while read -r n; do norm "$n"; done)"
for m in "${MODELS[@]}"; do
  if echo "$have" | grep -qxF "$(norm "$m")"; then
    ok "model $m present"
  else
    warn "pulling $m (first time only)..."
    ollama pull "$m"
  fi
done

# 3c. Warm models so the first real query is fast
if [ "$NO_WARM" = 0 ]; then
  info "Warming models (keep_alive 30m)..."
  curl -sf -m 120 "$OLLAMA/api/generate" \
    -d '{"model":"llama3.2:3b","prompt":"ok","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1
  curl -sf -m 60 "$OLLAMA/api/embed" \
    -d '{"model":"nomic-embed-text","input":"ok","keep_alive":"30m"}' >/dev/null 2>&1
  ok "models warm"
fi

# 4. Vector collection --------------------------------------------------
info "Checking Qdrant collection '$COLLECTION'..."
pts="$(curl -sf -m 5 "$QDRANT/collections/$COLLECTION" 2>/dev/null \
       | grep -oE '"points_count": *[0-9]+' | grep -oE '[0-9]+')"
if [ -n "$pts" ]; then
  ok "collection '$COLLECTION' exists ($pts points)"
else
  warn "collection '$COLLECTION' missing - run:  python -m src.ingestion.bootstrap_collection"
fi

echo
echo "Ready. If your shell isn't in the venv yet:  source venv/Scripts/activate"
echo
