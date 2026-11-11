#!/bin/zsh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$PROJECT_ROOT/packaging/macos-sanity"
APP="$PROJECT_ROOT/dist/CNP Function Explorer.app"

if [[ -n "${CNP_PYTHON:-}" && -x "${CNP_PYTHON:-}" ]]; then
    PYTHON_HINT="$CNP_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_HINT="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "/opt/miniconda3/envs/pycub310/bin/python" ]]; then
    PYTHON_HINT="/opt/miniconda3/envs/pycub310/bin/python"
else
    PYTHON_HINT="$(/usr/bin/which python3 2>/dev/null || true)"
fi

/bin/mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
/bin/cp "$SOURCE/Info.plist" "$APP/Contents/Info.plist"
/bin/cp "$SOURCE/PkgInfo" "$APP/Contents/PkgInfo"
/usr/bin/install -m 755 "$SOURCE/Sanity-CNP" "$APP/Contents/MacOS/Sanity-CNP"
/usr/bin/printf '%s\n' "$PYTHON_HINT" > "$APP/Contents/Resources/python-path"

/usr/bin/plutil -lint "$APP/Contents/Info.plist"
/usr/bin/codesign --force --deep --sign - "$APP"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"

echo "$APP"
