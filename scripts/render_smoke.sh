#!/usr/bin/env bash
set -euo pipefail

API_URL="${1:-${PARCAE_API_URL:-}}"
WEB_URL="${2:-${PARCAE_WEB_URL:-}}"
if [[ -z "$API_URL" ]]; then
  echo "Usage: $0 <api_base_url> [web_base_url]"
  echo "Example: $0 https://parcae-api.onrender.com https://parcae-web.onrender.com"
  exit 1
fi
API_URL="${API_URL%/}"
WEB_URL="${WEB_URL%/}"

tmp_html="$(mktemp)"
trap 'rm -f "$tmp_html"' EXIT

echo "[smoke] health"
curl -fsS "$API_URL/health" >/dev/null

echo "[smoke] capabilities"
CAPS="$(curl -fsS "$API_URL/ai/capabilities")"
echo "$CAPS"

for key in centurion_book_loaded centurion_tb_loaded centurion_nnue_loaded centurion_assets_ok centurion_strict_mode; do
  if ! printf "%s" "$CAPS" | grep -q "\"$key\":true"; then
    echo "[smoke] required capability is not true: $key"
    exit 1
  fi
done

echo "[smoke] create centurion match"
MATCH_JSON="$(curl -fsS -X POST "$API_URL/match" \
  -H "Content-Type: application/json" \
  -d '{"mode":"pva","ai_profiles":{"black":{"backend":"centurion","threads":1}}}')"
MATCH_ID="$(printf "%s" "$MATCH_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')"
echo "[smoke] match_id=$MATCH_ID"

echo "[smoke] step ai"
curl -fsS -X POST "$API_URL/match/$MATCH_ID/ai-step" >/dev/null

if [[ -n "$WEB_URL" ]]; then
  echo "[smoke] web root"
  curl -fsS "$WEB_URL/" >"$tmp_html"

  mapfile -t css_assets < <(grep -Eo 'href="/assets/[^"]+\.css"' "$tmp_html" | sed -E 's/^href="([^"]+)"$/\1/' | sort -u)
  mapfile -t js_assets < <(grep -Eo 'src="/assets/[^"]+\.js"' "$tmp_html" | sed -E 's/^src="([^"]+)"$/\1/' | sort -u)

  if [[ "${#css_assets[@]}" -eq 0 || "${#js_assets[@]}" -eq 0 ]]; then
    echo "[smoke] web assets missing in index.html"
    exit 1
  fi

  for asset in "${css_assets[@]}"; do
    headers="$(curl -fsSI "$WEB_URL$asset")"
    if ! printf "%s" "$headers" | tr '[:upper:]' '[:lower:]' | grep -q "content-type: text/css"; then
      echo "[smoke] bad css mime for $asset"
      echo "$headers"
      exit 1
    fi
  done

  for asset in "${js_assets[@]}"; do
    headers="$(curl -fsSI "$WEB_URL$asset")"
    if ! printf "%s" "$headers" | tr '[:upper:]' '[:lower:]' | grep -Eq "content-type: (application|text)/javascript"; then
      echo "[smoke] bad js mime for $asset"
      echo "$headers"
      exit 1
    fi
  done
fi

echo "[smoke] PASS"
