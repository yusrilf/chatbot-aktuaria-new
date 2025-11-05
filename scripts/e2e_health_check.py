#!/usr/bin/env python3
"""
E2E Health Check for Actuarial Chatbot
- Verifies /health, /api/storage/status, /documents/stats
- Logs timing, errors, and provides helpful fallbacks
"""
import argparse
import json
import sys
import time
from typing import Tuple

# Prefer requests, fallback to urllib
try:
    import requests
except Exception:
    requests = None
    import urllib.request
    import urllib.error


def fetch_json(url: str, timeout: float = 10.0) -> Tuple[int, dict]:
    start = time.time()
    try:
        if requests is not None:
            resp = requests.get(url, timeout=timeout)
            code = resp.status_code
            data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
        else:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                code = getattr(resp, 'status', 200)
                raw = resp.read().decode('utf-8')
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
        elapsed = time.time() - start
        return code, {"data": data, "elapsed": round(elapsed, 3)}
    except Exception as e:
        elapsed = time.time() - start
        return 0, {"error": str(e), "elapsed": round(elapsed, 3)}


def check_health(base_url: str) -> bool:
    code, payload = fetch_json(f"{base_url}/health")
    print(f"[HEALTH] code={code} elapsed={payload.get('elapsed')}s")
    if code != 200:
        print(f"[ERROR] Health endpoint returned {code}: {payload}")
        return False
    return True


def check_storage(base_url: str) -> bool:
    code, payload = fetch_json(f"{base_url}/api/storage/status")
    print(f"[STORAGE] code={code} elapsed={payload.get('elapsed')}s")
    if code != 200:
        print(f"[ERROR] Storage status returned {code}: {payload}")
        return False
    try:
        data = payload["data"].get("data", {})
        vs = data.get("vector_store", {})
        info = vs.get("collection_info", {})
        name = info.get("name")
        count = info.get("count")
        print(f"[STORAGE] vector_store status={vs.get('status')} name={name} count={count}")
        return bool(name) and count is not None
    except Exception as e:
        print(f"[ERROR] Parsing storage status failed: {e}")
        return False


def check_documents(base_url: str) -> bool:
    code, payload = fetch_json(f"{base_url}/documents/stats")
    print(f"[DOCS] code={code} elapsed={payload.get('elapsed')}s")
    if code != 200:
        print(f"[ERROR] Document stats returned {code}: {payload}")
        return False
    try:
        data = payload["data"].get("data", {})
        total = data.get("total_documents")
        print(f"[DOCS] total_documents={total} status={data.get('status')} collection={data.get('collection_info')}")
        return total is not None
    except Exception as e:
        print(f"[ERROR] Parsing document stats failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E health check")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Service base URL")
    parser.add_argument("--retries", type=int, default=3, help="Retries for flaky starts")
    args = parser.parse_args()

    success = True
    for attempt in range(1, args.retries + 1):
        print(f"\n[RUN] Attempt {attempt}/{args.retries} base={args.base_url}")
        h_ok = check_health(args.base_url)
        s_ok = check_storage(args.base_url)
        d_ok = check_documents(args.base_url)
        success = h_ok and s_ok and d_ok
        if success:
            break
        time.sleep(2)

    if not success:
        print("\n[DEBUG] Fallback tips:")
        print("- Ensure container exposes and maps port 5000")
        print("- Verify env keys: OPENAI_API_KEY, PINECONE_API_KEY, VECTOR_BACKEND=pinecone")
        print("- Check logs: docker logs <container>")
        return 1

    print("\n[SUCCESS] E2E health checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())