#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PHASE="${1:-}"
WITH_CLAUDE_CODE=false
for arg in "$@"; do
  if [ "$arg" = "--with-claude-code" ]; then
    WITH_CLAUDE_CODE=true
  fi
done

if [ -z "$PHASE" ]; then
  echo "Usage: $0 <phase-extra> [--with-claude-code]" >&2
  echo "  e.g.: $0 phase7" >&2
  echo "  Valid phase extras are the [project.optional-dependencies] groups in pyproject.toml:" >&2
  awk '/^\[project.optional-dependencies\]/{f=1;next}/^\[/{f=0}f && /^[a-zA-Z0-9_]+ = \[/{sub(/ = \[/,"");print "    "$0}' pyproject.toml >&2
  exit 1
fi

: "${RUNPOD_HOST:?Set RUNPOD_HOST in .env or the environment}"
: "${RUNPOD_PORT:?Set RUNPOD_PORT in .env or the environment}"
: "${RUNPOD_SSH_KEY:?Set RUNPOD_SSH_KEY in .env or the environment}"
: "${HF_TOKEN:?Set HF_TOKEN in .env or the environment}"

REMOTE_DIR="/workspace/warfarin-model-dev"
SSH_OPTS=(-i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" -o StrictHostKeyChecking=accept-new)

echo "==> Syncing model_dev/ to $RUNPOD_HOST:$REMOTE_DIR (excluding .venv, __pycache__, splits/, .env)"
ssh "${SSH_OPTS[@]}" "root@$RUNPOD_HOST" "mkdir -p $REMOTE_DIR"
tar --exclude='.venv' --exclude='__pycache__' --exclude='splits' --exclude='.env' --exclude='.git' \
    -czf - -C "$SCRIPT_DIR" . \
  | ssh "${SSH_OPTS[@]}" "root@$RUNPOD_HOST" "tar -xzf - -C $REMOTE_DIR"

echo "==> Also syncing the cleaned data this phase needs (data/eda_notebooks/cleaned)"
CLEANED_DIR="$SCRIPT_DIR/../data/eda_notebooks/cleaned"
if [ -d "$CLEANED_DIR" ]; then
  ssh "${SSH_OPTS[@]}" "root@$RUNPOD_HOST" "mkdir -p $REMOTE_DIR/../data/eda_notebooks/cleaned"
  tar -czf - -C "$CLEANED_DIR" . | ssh "${SSH_OPTS[@]}" "root@$RUNPOD_HOST" "tar -xzf - -C $REMOTE_DIR/../data/eda_notebooks/cleaned"
else
  echo "    (not found locally at $CLEANED_DIR — skipping; sync it manually if this pod needs real data)"
fi

echo "==> Installing uv and syncing dependencies for --extra $PHASE on the pod"
ssh "${SSH_OPTS[@]}" "root@$RUNPOD_HOST" bash -s <<REMOTE_SETUP
set -euo pipefail
cd "$REMOTE_DIR"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
uv sync --extra "$PHASE"

export HF_TOKEN="$HF_TOKEN"
echo "HF_TOKEN exported for this session (not persisted to disk on the pod)."
REMOTE_SETUP

if [ "$WITH_CLAUDE_CODE" = true ]; then
  echo "==> Installing Node + Claude Code CLI on the pod"
  ssh "${SSH_OPTS[@]}" "root@$RUNPOD_HOST" bash -s <<'REMOTE_CLAUDE'
set -euo pipefail
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
npm install -g @anthropic-ai/claude-code
REMOTE_CLAUDE
fi

cat <<EOF

==> Done. To connect:
    ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" root@$RUNPOD_HOST

==> Once connected:
    cd $REMOTE_DIR
    export HF_TOKEN=...    # re-export if you start a new shell session — it wasn't persisted
    uv run python phase0_scaffolding.py     # if not already run
    uv run python -m your_chosen_phase_entry_point

==> Before trusting any number this pod produces: add it to RESULTS.md with a
    real evidence path, per RESULTS.md's own rule. A number that only exists
    in a terminal scrollback or a chat message doesn't count.
EOF
