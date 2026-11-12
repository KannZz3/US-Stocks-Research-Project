#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import streamlit, pandas, numpy, plotly' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
print "Starting dashboard. Keep this window open; Ctrl+C stops it."
exec .venv/bin/python -m streamlit run app.py --server.headless false
