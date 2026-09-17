#!/usr/bin/env bash
# Prepare a verl checkout for the EPLB recipe.
#
# The recipe targets public verl at the commit recorded in REQUIRED_VERL.txt and
# applies patches/verl-eplb-core.patch on top of it. That patch carries the EPLB
# training/rollout implementation this recipe drives; the recipe itself holds the
# orchestration, strategy, config and entrypoint layers.
#
#   VERL_SOURCE=/path/to/verl bash scripts/setup_verl.sh
#   VERL_SOURCE=/path/to/verl SKIP_INSTALL=1 bash scripts/setup_verl.sh   # patch only
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Strip whitespace: a CRLF checkout of REQUIRED_VERL.txt otherwise leaves a
# trailing CR on the SHA and the comparison below fails with two identical
# looking strings. .gitattributes pins LF, this is the belt to that braces.
expected="$(awk -F= '/^VERL_COMMIT=/{print $2}' "$recipe_dir/REQUIRED_VERL.txt" | tr -d '[:space:]')"
patch="$recipe_dir/patches/verl-eplb-core.patch"

: "${VERL_SOURCE:?Set VERL_SOURCE to a verl checkout at $expected}"

actual="$(git -C "$VERL_SOURCE" rev-parse HEAD)"
if [[ "$actual" != "$expected" ]]; then
  echo "Expected verl $expected, found $actual." >&2
  echo "Check out the pinned commit first:" >&2
  echo "  git -C $VERL_SOURCE fetch https://github.com/verl-project/verl.git $expected" >&2
  echo "  git -C $VERL_SOURCE checkout $expected" >&2
  exit 1
fi

# The patch is stored with LF endings. A checkout made under core.autocrlf=true
# rewrites the working tree to CRLF and every hunk then fails on context, so the
# apply is pinned to the same normalisation the patch was generated under.
git_apply=(git -c core.autocrlf=false -C "$VERL_SOURCE" apply)

if "${git_apply[@]}" --reverse --check "$patch" 2>/dev/null; then
  echo "EPLB core patch is already applied."
elif "${git_apply[@]}" --check "$patch" 2>/dev/null; then
  "${git_apply[@]}" "$patch"
  echo "Applied $(basename "$patch")."
else
  echo "EPLB core patch conflicts with the supplied checkout:" >&2
  "${git_apply[@]}" --check "$patch" || true
  exit 1
fi

if [[ -n "${SKIP_INSTALL:-}" ]]; then
  echo "SKIP_INSTALL set; patched the checkout without installing."
  exit 0
fi

# Keep the existing accelerator environment intact; build/install the patched
# library without resolving or replacing torch, torch_npu, CANN or other deps.
uv pip install --python "${PYTHON:-python3}" --no-deps "$VERL_SOURCE"
