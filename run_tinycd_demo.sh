#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="$root/.venv/bin/python"
packages="$root/artifacts/tinycd/streamlit-deps"

if [ ! -x "$python" ]; then
    printf 'Proje Python ortamı bulunamadı: %s\n' "$python" >&2
    exit 1
fi
if [ -d "$packages" ]; then
    export PYTHONPATH="$packages${PYTHONPATH:+:$PYTHONPATH}"
fi
if ! "$python" -B -c 'import streamlit' 2>/dev/null; then
    printf 'Streamlit yüklenemedi. Kurulum için %s dosyasına bak.\n' "$root/docs/tinycd_demo.md" >&2
    exit 1
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="$root/artifacts/tinycd/mpl"
mkdir -p "$MPLCONFIGDIR"
cd "$root"

exec "$python" -B -m streamlit run streamlit_tinycd.py \
    --global.developmentMode false \
    --server.address 127.0.0.1 \
    --server.headless true \
    --client.toolbarMode minimal \
    --browser.gatherUsageStats false
