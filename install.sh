#!/usr/bin/env bash
# graft from source (default: the main branch): curl -fsSL https://raw.githubusercontent.com/csabesz1130/graft.sh/main/install.sh | bash
# GRAFT_REF picks a branch or tag. GRAFT_EXTRAS=train also installs torch, transformers and peft for the built-in trainer.
set -euo pipefail
REF="${GRAFT_REF:-main}"
EXTRAS="${GRAFT_EXTRAS:-}"
SOURCE="git+https://github.com/csabesz1130/graft.sh@${REF}"
SPEC="graft-sh${EXTRAS:+[$EXTRAS]} @ ${SOURCE}"
die() { echo "error: $*" >&2; exit 1; }
command -v git >/dev/null || die "git is required (sudo apt install git)"
echo "installing graft from ${REF} ..."
if command -v uv >/dev/null; then
  uv tool install --force --python ">=3.11" "${SPEC}"
elif command -v pipx >/dev/null; then
  pipx install --force "${SPEC}"
else
  PY="$(command -v python3 || true)"
  [ -n "${PY}" ] || die "python 3.11 or newer is required"
  "${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "python 3.11 or newer is required, found $("${PY}" --version)"
  "${PY}" -m pip install --user --upgrade "${SPEC}"
fi
if command -v graft >/dev/null; then
  graft --version
else
  echo "installed. add ~/.local/bin to your PATH to run it: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
