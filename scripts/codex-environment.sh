#!/usr/bin/env bash
set -euo pipefail

mode="${1:-setup}"
if [[ "$mode" != "setup" && "$mode" != "maintenance" ]]; then
  printf 'usage: %s [setup|maintenance]\n' "$0" >&2
  exit 2
fi

root="$(git rev-parse --show-toplevel)"
parent="$(dirname "$root")"
foundation_dir="$parent/moenarch-foundation"
tooling_dir="$parent/coding-tooling"
tooling_rev="25cdc38e079a959821317cbf450f2c96e030ce3c"
onnxruntime_version="1.24.4"
venv_dir="$HOME/.codex-venvs/visual-analysis"

log() {
  printf '[visual-analysis codex] %s\n' "$*"
}

run_privileged() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

if [[ "$mode" == "setup" ]] && command -v apt-get >/dev/null 2>&1; then
  log "installing native visual-analysis prerequisites"
  run_privileged apt-get update
  run_privileged apt-get install -y --no-install-recommends \
    ca-certificates \
    clang \
    cmake \
    curl \
    ffmpeg \
    git \
    libssl-dev \
    pkg-config \
    poppler-utils \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng
fi

desired_bun="$(python3 - <<'PY'
import json
from pathlib import Path
value = json.loads(Path('package.json').read_text())['packageManager']
name, version = value.split('@', 1)
assert name == 'bun'
print(version)
PY
)"

if ! command -v bun >/dev/null 2>&1 || [[ "$(bun --version)" != "$desired_bun" ]]; then
  log "installing Bun $desired_bun"
  curl -fsSL https://bun.sh/install | bash -s "bun-v${desired_bun}"
fi
export PATH="$HOME/.bun/bin:$PATH"

rust_toolchain="$(python3 - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path('rust-toolchain.toml').read_text())['toolchain']['channel'])
PY
)"
log "ensuring Rust $rust_toolchain with rustfmt and clippy"
rustup toolchain install "$rust_toolchain" --component rustfmt --component clippy --profile minimal

if [[ "$mode" == "setup" || ! -x "$venv_dir/bin/python" ]]; then
  log "preparing ONNX Runtime $onnxruntime_version"
  python3 -m venv "$venv_dir"
  "$venv_dir/bin/pip" install --disable-pip-version-check --upgrade pip
  "$venv_dir/bin/pip" install --disable-pip-version-check "onnxruntime==$onnxruntime_version"
fi

ort_dylib="$($venv_dir/bin/python - <<'PY'
from pathlib import Path
import onnxruntime
capi = Path(onnxruntime.__file__).parent / 'capi'
candidates = sorted(capi.glob('libonnxruntime.so*'))
if not candidates:
    raise SystemExit('onnxruntime Python package did not contain libonnxruntime.so')
print(candidates[-1].resolve())
PY
)"
export ORT_DYLIB_PATH="$ort_dylib"

read -r foundation_git foundation_rev < <(python3 - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('.coding-tooling.source-deps.json').read_text())
patches = config['cargo']['patches']
gits = {patch['git'] for patch in patches}
revs = {patch['rev'] for patch in patches}
if len(gits) != 1 or len(revs) != 1:
    raise SystemExit('Codex environment expects exactly one pinned source repository')
print(next(iter(gits)), next(iter(revs)))
PY
)

sync_checkout() {
  local url="$1"
  local revision="$2"
  local destination="$3"

  if [[ ! -d "$destination/.git" ]]; then
    log "creating $(basename "$destination") checkout"
    rm -rf "$destination"
    git init -q "$destination"
    git -C "$destination" remote add origin "$url"
  fi

  git -C "$destination" fetch -q --depth=1 origin "$revision"
  git -C "$destination" reset -q --hard
  git -C "$destination" clean -q -fd
  git -C "$destination" checkout -q --detach "$revision"
}

log "syncing exact foundation source revision $foundation_rev"
sync_checkout "$foundation_git" "$foundation_rev" "$foundation_dir"

log "syncing pinned coding-tooling revision $tooling_rev"
sync_checkout "https://github.com/moritzbrantner/coding-tooling" "$tooling_rev" "$tooling_dir"
export CODING_TOOLING_DIR="$tooling_dir"

python3 - "$HOME/.bashrc" "$tooling_dir" "$ort_dylib" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
tooling_dir = sys.argv[2]
ort_dylib = sys.argv[3]
start = '# visual-analysis codex environment:start'
end = '# visual-analysis codex environment:end'
text = path.read_text() if path.exists() else ''
if start in text and end in text:
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    text = before.rstrip() + '\n' + after.lstrip('\n')
block = f'''{start}
export PATH="$HOME/.bun/bin:$PATH"
export CODING_TOOLING_DIR="{tooling_dir}"
export ORT_DYLIB_PATH="{ort_dylib}"
{end}
'''
path.write_text(text.rstrip() + '\n\n' + block)
PY

log "installing pinned Bun dependencies"
bun install --cwd "$tooling_dir" --frozen-lockfile
bun install --cwd "$root" --frozen-lockfile

log "activating exact source-development graph"
bash "$root/scripts/source-deps" activate

log "warming Cargo dependency cache"
cargo fetch --manifest-path "$root/Cargo.toml"

log "running environment preflight"
cargo metadata --manifest-path "$root/Cargo.toml" --format-version 1 --no-deps >/dev/null
bun run --cwd "$root" structural:prepare
"$venv_dir/bin/python" -c 'import onnxruntime; print("onnxruntime", onnxruntime.__version__)'
tesseract --version | head -n 1
ffmpeg -version | head -n 1
pdftotext -v 2>&1 | head -n 1

log "$mode complete"
