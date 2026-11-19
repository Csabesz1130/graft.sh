#!/usr/bin/env bash
# Test graft end to end on a fresh Linux machine. Run it from a checkout of this repository:
#   scripts/server-test.sh              # unit tests, a clean install, the demo rebase, its alternate endings
#   scripts/server-test.sh --train      # also installs torch, transformers and peft and runs the trainer tests
# Everything happens in a temporary folder and virtualenv; nothing is installed system-wide. Exit 0 = all passed.
set -uo pipefail
TRAIN=0
[ "${1:-}" = "--train" ] && TRAIN=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
PORT="${GRAFT_TEST_PORT:-8811}"
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
PY="$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3 || true)"
[ -n "${PY}" ] || { echo "python 3.11 or newer is required"; exit 2; }
"${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "python 3.11 or newer is required, found $("${PY}" --version)"; exit 2; }
echo "python: $("${PY}" --version)"

step "clean install into a fresh virtualenv"
"${PY}" -m venv "${WORK}/venv" || { echo "cannot create a virtualenv (apt install python3-venv)"; exit 2; }
VPY="${WORK}/venv/bin/python"; GRAFT="${WORK}/venv/bin/graft"
cp -r "${REPO}" "${WORK}/src" && rm -rf "${WORK}/src/.git" "${WORK}/src/.graft"
run "pip install ." "${VPY}" -m pip install -q "${WORK}/src"
run "graft --version" "${GRAFT}" --version

if [ "${TRAIN}" = 1 ]; then
  step "training dependencies"
  run "pip install .[train]" "${VPY}" -m pip install -q "${WORK}/src[train]"
  "${VPY}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
fi

step "unit and end-to-end tests"
( cd "${WORK}/src" && "${VPY}" -m unittest discover -s tests -t . 2>&1 | tail -n 6 ) | tee "${WORK}/tests.log"
grep -q '^OK' "${WORK}/tests.log" || FAILED+=("unit tests")
[ "${TRAIN}" = 1 ] && { grep -q 'skipped' "${WORK}/tests.log" && { echo "trainer tests were skipped even with --train"; FAILED+=("trainer tests skipped"); }; }

step "the demo, as a user would run it"
cp -r "${WORK}/src/examples/support-desk" "${WORK}/demo"
cd "${WORK}/demo" || exit 2
"${VPY}" mock_models.py --port "${PORT}" >"${WORK}/mock.log" 2>&1 & MOCK_PID=$!
for _ in $(seq 100); do curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break; sleep 0.1; done
sed -i "s/127.0.0.1:8811/127.0.0.1:${PORT}/g; s#\"python3\"#\"${VPY}\"#g" graft.toml
sed -i "s#\"python3\"#\"${VPY}\"#g" genome/rules/*.toml
export PATH="${WORK}/venv/bin:${PATH}"
run "graft models"            graft models
run "graft check --model prod" graft check --model prod
graft verify new-base >/dev/null 2>&1 && { echo "FAIL  verify new-base should fail before a rebase"; FAILED+=("verify before rebase"); } || echo "ok    verify new-base fails before the rebase (exit 1, as expected)"
graft rebase --target new-base >"${WORK}/rebase.log" 2>&1; code=$?
cat "${WORK}/rebase.log"
[ "${code}" = 0 ] && grep -q "52/52 cases   5 of 5 rules hold" "${WORK}/rebase.log" && echo "ok    full rebase" || { echo "FAIL  full rebase (exit ${code})"; FAILED+=("full rebase"); }
run "graft show"              graft show
run "graft verify --graft"    graft verify new-base --graft
run "graft log"               graft log
rm -rf .graft
graft rebase --target new-base --ladder zero-shot,prompt >"${WORK}/conflict.log" 2>&1; code=$?
[ "${code}" = 1 ] && grep -q "CONFLICT: 12 cases" "${WORK}/conflict.log" && echo "ok    conflict ending (exit 1)" || { echo "FAIL  conflict ending (exit ${code})"; FAILED+=("conflict ending"); }
run "continue to a pass"      graft rebase --continue --ladder zero-shot,prompt,lora
rm -rf .graft
run "fast-forward"            graft rebase --target prod

step "result"
if [ "${#FAILED[@]}" = 0 ]; then echo "ALL PASSED"; exit 0; fi
printf 'FAILED: %s\n' "${FAILED[@]}"; exit 1
