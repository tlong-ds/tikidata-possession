#!/usr/bin/env bash
# Builds the upload pair <team>.csv + <team>.pdf in submission/ and validates the exact CSV to upload.
#   ./build.sh                 # TikiData and members as written in technical.tex
#   ./build.sh "TEAM_NAME" "Member A, Member B"
set -euo pipefail
cd "$(dirname "$0")"
TEAM="${1:-TikiData}"
MEMBERS="${2:-}"
tex() { printf '%s' "$1" | sed -e 's/\\/\\textbackslash{}/g' -e 's/[_&%$#{}]/\\&/g'; }

SRC="../$TEAM.csv"
[ -f "$SRC" ] || SRC=../evaluation/results/nap3.csv
[ -f "$SRC" ] || { echo "Missing CSV: run python -m src.submit run --model final_metrica --check-twice --out $TEAM.csv first"; exit 1; }
cp "$SRC" "$TEAM.csv"
python ../evaluation/validate_submission.py "$TEAM.csv"
echo "sha256 $(shasum -a 256 "$TEAM.csv" | cut -d' ' -f1)"

DEFS="\\def\\teamname{$(tex "$TEAM")}"
[ -n "$MEMBERS" ] && DEFS="$DEFS\\def\\members{$(tex "$MEMBERS")}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$TEAM" \
  -pdflatex="pdflatex %O '$DEFS\input{%S}'" \
  technical.tex >/dev/null
latexmk -c -jobname="$TEAM" technical.tex >/dev/null
echo "Created submission/$TEAM.csv and submission/$TEAM.pdf"
