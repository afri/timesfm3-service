#!/usr/bin/env bash
set -e

PORT="${1:-8000}"
BASE_URL="http://localhost:${PORT}"

echo "Checking Health..."
curl -s "${BASE_URL}/health" | jq . || curl -s "${BASE_URL}/health"
echo -e "\n"

echo "Sending Forecast Request..."
curl -s -X POST "${BASE_URL}/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [10.5, 11.2, 12.0, 11.8, 12.5, 13.1, 13.0, 13.8, 14.2, 14.9],
    "horizon": 5,
    "return_quantiles": true
  }' | jq . || curl -s -X POST "${BASE_URL}/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [10.5, 11.2, 12.0, 11.8, 12.5, 13.1, 13.0, 13.8, 14.2, 14.9],
    "horizon": 5,
    "return_quantiles": true
  }'
echo -e "\n"
