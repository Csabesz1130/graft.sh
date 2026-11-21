#!/usr/bin/env bash
# Test graft end to end on a fresh Linux machine. Run it from a checkout of this repository:
#   scripts/server-test.sh              # unit tests, a clean install, the demo rebase, its alternate endings
#   scripts/server-test.sh --train      # also installs torch, transformers and peft and runs the trainer tests
# Everything happens in a temporary folder and virtualenv; nothing is installed system-wide. Exit 0 = all passed.
# Set GRAFT_TEST_PYTHON to select an interpreter. Without an NVIDIA GPU, --train installs torch from the CPU wheel index;
# GRAFT_TORCH_INDEX_URL picks another index (empty: the default CUDA build).
set -uo pipefail
TRAIN=0
if [[ "$#" == 1 && "$1" == "--train" ]]; then TRAIN=1
elif [[ "$#" != 0 ]]; then echo "usage: scripts/server-test.sh [--train]" >&2; exit 2; fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)" || exit 2
PORT="${GRAFT_TEST_PORT:-8811}"
if ! [[ "${PORT}" =~ ^[1-9][0-9]{0,4}$ ]] || (( PORT > 65535 )); then
  echo "GRAFT_TEST_PORT must be a port from 1 to 65535" >&2; rm -rf "${WORK}"; exit 2
fi
MOCK_PID=""
FAILED=()
cleanup() { [ -n "${MOCK_PID}" ] && kill "${MOCK_PID}" 2>/dev/null; rm -rf "${WORK}"; }
trap cleanup EXIT

step() { printf '\n== %s\n' "$1"; }
run() {  # run <name> <command...>: log the result, keep going
  local name="$1"; shift
  if "$@" >"${WORK}/last.log" 2>&1; then echo "ok    ${name}"; else
    echo "FAIL  ${name}"; tail -n 15 "${WORK}/last.log" | sed 's/^/      /'; FAILED+=("${name}"); fi
}

step "machine"
echo "$(uname -sm), $(nproc) cores, $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB RAM"}')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no NVIDIA GPU"
if [ -n "${GRAFT_TEST_PYTHON:-}" ]; then
  PY="$(command -v -- "${GRAFT_TEST_PYTHON}")" || { echo "GRAFT_TEST_PYTHON not found" >&2; exit 2; }
else
  PY=""
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null && "${candidate}" -c 'import ensurepip, venv' 2>/dev/null; then
      PY="$(command -v "${candidate}")"; break
    fi
  done
fi
[ -n "${PY}" ] || { echo "python 3.11 or newer is required"; exit 2; }
"${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "python 3.11 or newer is required, found $("${PY}" --version)"; exit 2; }
echo "python: $("${PY}" --version)"

step "clean install into a fresh virtualenv"
"${PY}" -m venv "${WORK}/venv" || { echo "cannot create a virtualenv (apt install python3-venv)"; exit 2; }
VPY="${WORK}/venv/bin/python"; GRAFT="${WORK}/venv/bin/graft"
mkdir "${WORK}/src" || exit 2
tar -C "${REPO}" --exclude-vcs --exclude=.graft --exclude=.venv --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude=.ruff_cache --exclude='*.pyc' -cf - \
  LICENSE README.md pyproject.toml graft examples scripts tests | tar -xf - -C "${WORK}/src" || exit 2
run "pip install ." "${VPY}" -m pip install -q "${WORK}/src"
run "graft --version" "${GRAFT}" --version

if [ "${TRAIN}" = 1 ]; then
  step "training dependencies"
  # No NVIDIA GPU: the CPU wheels (about 200 MB) instead of the default CUDA build (about 5 GB of wheels this host cannot use).
  # GRAFT_TORCH_INDEX_URL overrides this; set it empty to install the default build anyway.
  if [ -z "${GRAFT_TORCH_INDEX_URL+set}" ] && ! command -v nvidia-smi >/dev/null; then
    GRAFT_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
  fi
  echo "torch from ${GRAFT_TORCH_INDEX_URL:-the default index}"
  if [ -n "${GRAFT_TORCH_INDEX_URL:-}" ]; then
    "${VPY}" -m pip install -q --index-url "${GRAFT_TORCH_INDEX_URL}" 'torch>=2.2' || exit 1
  fi
  run "pip install .[train]" "${VPY}" -m pip install -q "${WORK}/src[train]"
  "${VPY}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
fi

step "unit and end-to-end tests"
CHECKS="${WORK}/src/scripts/server_checks.py"
suite() {
  cd "${WORK}/src" || return 1
  if [ "${TRAIN}" = 1 ]; then
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
      "${VPY}" "${CHECKS}" tests --train
  else
    "${VPY}" "${CHECKS}" tests
  fi
}
( suite ) 2>&1 | tee "${WORK}/tests.log" || FAILED+=("unit and trainer tests")

step "the demo, as a user would run it"
cp -r "${WORK}/src/examples/support-desk" "${WORK}/demo"
cd "${WORK}/demo" || exit 2
# The private marker is created only by our child after it owns the listening socket.
"${VPY}" -c '
import sys
from pathlib import Path
from mock_models import serve
with serve(int(sys.argv[1])) as server:
    Path(sys.argv[2]).touch(exist_ok=False)
    server.serve_forever()
' "${PORT}" "${WORK}/mock.ready" >"${WORK}/mock.log" 2>&1 & MOCK_PID=$!
ready=0
for _ in $(seq 100); do
  kill -0 "${MOCK_PID}" 2>/dev/null || break
  if [ -e "${WORK}/mock.ready" ] && curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 0.1
done
if [ "${ready}" != 1 ]; then echo "mock server failed to start" >&2; cat "${WORK}/mock.log" >&2; exit 1; fi
sed -i "s/127.0.0.1:8811/127.0.0.1:${PORT}/g" graft.toml
export PATH="${WORK}/venv/bin:${PATH}"
run "graft models"             "${GRAFT}" models
run "graft check --model prod" "${GRAFT}" check --model prod
run "verify before rebase returns exit 1 with 13/52" "${VPY}" "${CHECKS}" before
"${GRAFT}" rebase --target new-base 2>&1 | tee "${WORK}/rebase.log" || FAILED+=("full rebase")
run "52/52 cases, 5/5 rules, verified adapter" "${VPY}" "${CHECKS}" full
run "graft show"               "${GRAFT}" show
run "graft log"                "${GRAFT}" log
rm -rf .graft
conflict() {
  local code=0
  "${GRAFT}" rebase --target new-base --ladder zero-shot,prompt || code=$?
  [ "${code}" = 1 ] || { echo "expected conflict exit 1, got ${code}"; return 1; }
  "${VPY}" "${CHECKS}" conflict
}
run "conflict, 12 unresolved cases" conflict
run "continue to a pass"       "${GRAFT}" rebase --continue --ladder zero-shot,prompt,lora
run "continued adapter verifies" "${VPY}" "${CHECKS}" continue
rm -rf .graft
run "fast-forward"             "${GRAFT}" rebase --target prod
run "fast-forward without an adapter" "${VPY}" "${CHECKS}" fast-forward

step "result"
if [ "${#FAILED[@]}" = 0 ]; then echo "ALL PASSED"; exit 0; fi
printf 'FAILED: %s\n' "${FAILED[@]}"; exit 1
