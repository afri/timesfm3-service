"""Sample Python client demonstrating TimesFM 3 service interaction."""

import json
import urllib.request
import sys

PORT = sys.argv[1] if len(sys.argv) > 1 else 8000
BASE_URL = f"http://localhost:{PORT}"


def test_forecast():
    url = f"{BASE_URL}/v1/forecast"
    payload = {
        "series": [
            [1.2, 2.5, 3.1, 4.8, 5.0, 6.2, 7.1],
            [10.0, 9.5, 9.0, 8.2, 7.5, 7.0, 6.4],
        ],
        "horizon": 6,
        "return_quantiles": True,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        result = json.loads(body)
        print("Forecast response received:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        test_forecast()
    except Exception as e:
        print(f"Error communicating with service: {e}")
        sys.exit(1)
