#!/usr/bin/env bash

set -euo pipefail

# Download target directory (default: ./data)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${1:-${PROJECT_ROOT}/data}"

mkdir -p "${OUT_DIR}"

download_with_gdown() {
  local file_id="$1"
  local out_path="$2"
  gdown --id "${file_id}" --output "${out_path}" --fuzzy
}

download_with_curl() {
  local file_id="$1"
  local out_path="$2"
  local cookie_file
  local confirm_token

  cookie_file="$(mktemp)"
  trap 'rm -f "${cookie_file}"' RETURN

  curl -s -L -c "${cookie_file}" \
    "https://drive.google.com/uc?export=download&id=${file_id}" \
    -o /tmp/gdrive_page.html

  confirm_token="$(sed -n 's/.*confirm=\([0-9A-Za-z_]\+\).*/\1/p' /tmp/gdrive_page.html | head -n 1 || true)"

  if [[ -n "${confirm_token}" ]]; then
    curl -L -b "${cookie_file}" \
      "https://drive.google.com/uc?export=download&confirm=${confirm_token}&id=${file_id}" \
      -o "${out_path}"
  else
    # Small files may not require confirmation.
    curl -L \
      "https://drive.google.com/uc?export=download&id=${file_id}" \
      -o "${out_path}"
  fi

  rm -f /tmp/gdrive_page.html
}

download_file() {
  local file_id="$1"
  local filename="$2"
  local out_path="${OUT_DIR}/${filename}"

  echo "Downloading ${filename} ..."
  if command -v gdown >/dev/null 2>&1; then
    download_with_gdown "${file_id}" "${out_path}"
  else
    download_with_curl "${file_id}" "${out_path}"
  fi
  echo "Saved: ${out_path}"
}

# File IDs from:
# toys.zip   -> https://drive.google.com/file/d/14yHjyJSyzZRy7z-uzjHmukMh3-0LpATY/view?usp=sharing
# sports.zip -> https://drive.google.com/file/d/1v6EUEZAou_AIM0rk1JrJdBKXjSOmqxoM/view?usp=sharing
# beauty.zip -> https://drive.google.com/file/d/1-Q7wMpxVGeZxo3TwW-eCd63Wi6APEiZz/view?usp=sharing
download_file "14yHjyJSyzZRy7z-uzjHmukMh3-0LpATY" "toys.zip"
download_file "1v6EUEZAou_AIM0rk1JrJdBKXjSOmqxoM" "sports.zip"
download_file "1-Q7wMpxVGeZxo3TwW-eCd63Wi6APEiZz" "beauty.zip"

echo "All downloads completed."
