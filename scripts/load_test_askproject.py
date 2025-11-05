#!/usr/bin/env python3
"""Load Test: Send 100 short POST requests to /askproject.

- Runs sequential requests with simple payloads to exercise normal flow.
- Logs status, latency, and aggregates a summary.
- Fetches conversation history for the test session at the end.

Designed for macOS and Python 3.
"""
import os
import time
import logging
import statistics
import random
from typing import Dict, Any, List

import requests

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("load_test_askproject")

REQUEST_COUNT = int(os.getenv("LOADTEST_COUNT", "100"))
SESSION_ID = os.getenv("LOADTEST_SESSION", "loadtest")

# Resolve port with fallback chain similar to app/main.py
PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5001")))
BASE_URL = f"http://localhost:{PORT}"
ASKPROJECT_URL = f"{BASE_URL}/askproject"
HISTORY_URL = f"{BASE_URL}/conversation/history"

# Payload generator
SHORT_TEMPLATES = [
    "tes singkat {i}",
    "halo {i}",
    "cek {i}",
    "uji {i}",
    "ping {i}",
    "ok {i}",
]

def make_payload(i: int) -> Dict[str, Any]:
    text = random.choice(SHORT_TEMPLATES).format(i=i)
    return {"question": text, "session_id": SESSION_ID}


def run_load_test() -> Dict[str, Any]:
    latencies: List[float] = []
    successes = 0
    failures: List[str] = []

    logger.info(f"Starting load test: {REQUEST_COUNT} requests -> {ASKPROJECT_URL}")

    session = requests.Session()
    timeout_s = float(os.getenv("LOADTEST_TIMEOUT", "60"))

    for i in range(1, REQUEST_COUNT + 1):
        payload = make_payload(i)
        start = time.time()
        try:
            resp = session.post(ASKPROJECT_URL, json=payload, timeout=timeout_s)
            elapsed = time.time() - start
            latencies.append(elapsed)

            # Basic success criteria: HTTP 200 and success flag
            is_ok = False
            try:
                data = resp.json()
                is_ok = (resp.status_code == 200) and bool(data.get("success", False))
            except Exception:
                data = {"raw": resp.text}

            if is_ok:
                successes += 1
                if i % 10 == 0:
                    logger.info(f"[{i}/{REQUEST_COUNT}] OK in {elapsed:.2f}s")
            else:
                msg = f"[{i}/{REQUEST_COUNT}] FAIL status={resp.status_code} body={str(data)[:200]}"
                failures.append(msg)
                logger.warning(msg)
        except requests.RequestException as e:
            elapsed = time.time() - start
            latencies.append(elapsed)
            msg = f"[{i}/{REQUEST_COUNT}] ERROR {type(e).__name__}: {e}"
            failures.append(msg)
            logger.error(msg)
            # Fallback: brief backoff to allow server recovery if needed
            time.sleep(0.2)

        # Small pacing to avoid tight loop hammering
        time.sleep(0.05)

    summary = summarize_results(latencies, successes, failures)
    return summary


def summarize_results(latencies: List[float], successes: int, failures: List[str]) -> Dict[str, Any]:
    count = len(latencies)
    avg = statistics.mean(latencies) if latencies else 0.0
    p95 = (statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0.0)
    mn = min(latencies) if latencies else 0.0
    mx = max(latencies) if latencies else 0.0

    summary = {
        "request_count": count,
        "successes": successes,
        "failures": len(failures),
        "latency_sec": {
            "avg": round(avg, 3),
            "p95": round(p95, 3),
            "min": round(mn, 3),
            "max": round(mx, 3),
        },
        "session_id": SESSION_ID,
    }

    logger.info(f"Summary: {summary}")
    if failures:
        logger.info("Sample failures (up to 5):")
        for line in failures[:5]:
            logger.info(line)

    return summary


def fetch_history(limit: int = 10) -> Dict[str, Any]:
    params = {"session_id": SESSION_ID, "limit": str(limit)}
    try:
        resp = requests.get(HISTORY_URL, params=params, timeout=30)
        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {"raw": resp.text}
        logger.info(f"History status={resp.status_code} items={limit}")
        return data
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return {"success": False, "message": str(e)}


def main():
    # Quick connectivity check
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        logger.info(f"Root check: status={r.status_code}")
    except Exception as e:
        logger.error(f"Server not reachable at {BASE_URL}: {e}")
        logger.error("Ensure the Flask server is running before starting the load test.")
        return

    summary = run_load_test()
    history = fetch_history(limit=10)

    print("\n===== LOAD TEST SUMMARY =====")
    print(summary)
    print("\n===== HISTORY (limit=10) =====")
    print(str(history)[:1500])


if __name__ == "__main__":
    main()