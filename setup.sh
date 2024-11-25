#!/bin/bash

# Parallel Global Voices Dataset
URL="https://nlp.ilsp.gr/pgv/meta/docpairs.txt.zip"
ENG_SPA="https://nlp.ilsp.gr/pgv/archives/gv-eng-20041026-3.xml"
ZIP_FILE="gv-eng-20041026-3.xml"
DEST_DIR="datasets"

curl -O "$ENG_SPA"
mkdir -p "$DEST_DIR"

unzip "$ZIP_FILE" -d "$DEST_DIR"
rm "$ZIP_FILE"
echo "PGV: Download and extraction complete, ZIP file deleted."
