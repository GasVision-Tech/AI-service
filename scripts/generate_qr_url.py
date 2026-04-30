"""Utility script for generating signed QR entry URLs.

Example:
    python generate_qr_url.py --station-code AZS-001 --base-url http://localhost:8089
"""

from __future__ import annotations

import argparse
import os

from itsdangerous import URLSafeTimedSerializer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-code", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--secret", default=os.getenv("QR_ACCESS_SECRET"))
    args = parser.parse_args()
    if not args.secret:
        raise SystemExit("QR secret is required: pass --secret or set QR_ACCESS_SECRET")

    serializer = URLSafeTimedSerializer(args.secret, salt="gasvision-qr")
    token = serializer.dumps({"station_code": args.station_code})
    print(f"{args.base_url}?access_token={token}")


if __name__ == "__main__":
    main()
