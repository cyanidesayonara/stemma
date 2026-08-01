"""Commit a pending Partner Center submission after updateMetadata."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://manage.devcenter.microsoft.com/v1.0"
TOKEN_SCOPE = "https://manage.devcenter.microsoft.com/.default"
WAIT_STATUSES = frozenset({"PendingCommit", "CommitStarted"})


def _lookup(mapping: dict, *keys: str):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _request(
    method: str,
    url: str,
    token: str,
    *,
    payload: dict | None = None,
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {detail}") from exc


def _get_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    data = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": TOKEN_SCOPE,
            "grant_type": "client_credentials",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data=data,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        token_payload = json.load(resp)
    token = token_payload.get("access_token")
    if not token:
        raise RuntimeError("Azure AD token response missing access_token")
    return token


def _pending_submission_id(application: dict) -> str | None:
    pending = _lookup(
        application,
        "pendingApplicationSubmission",
        "PendingApplicationSubmission",
    )
    if not isinstance(pending, dict):
        return None
    submission_id = _lookup(pending, "id", "Id")
    return str(submission_id) if submission_id else None


def _poll_commit(
    token: str,
    product_id: str,
    submission_id: str,
    *,
    timeout_s: int,
    interval_s: int,
    verbose: bool,
) -> dict:
    deadline = time.monotonic() + timeout_s
    status_url = (
        f"{BASE_URL}/my/applications/{product_id}/submissions/{submission_id}/status"
    )
    last: dict = {}
    while time.monotonic() < deadline:
        last = _request("GET", status_url, token)
        status = _lookup(last, "status", "Status")
        if verbose:
            print(f"Submission status: {status}")
        if status not in WAIT_STATUSES:
            errors = _lookup(last, "statusDetails", "StatusDetails") or {}
            error_items = _lookup(errors, "errors", "Errors") or []
            if error_items:
                raise RuntimeError(f"Submission commit failed: {error_items}")
            return last
        time.sleep(interval_s)
    raise TimeoutError(
        f"Timed out waiting for submission commit after {timeout_s}s; last={last!r}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-id", default=os.environ.get("PRODUCT_ID"))
    parser.add_argument(
        "--tenant-id",
        default=os.environ.get("PARTNER_CENTER_TENANT_ID"),
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("PARTNER_CENTER_CLIENT_ID"),
    )
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("PARTNER_CENTER_CLIENT_SECRET"),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    missing = [
        name
        for name, value in (
            ("PRODUCT_ID", args.product_id),
            ("PARTNER_CENTER_TENANT_ID", args.tenant_id),
            ("PARTNER_CENTER_CLIENT_ID", args.client_id),
            ("PARTNER_CENTER_CLIENT_SECRET", args.client_secret),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required setting(s): {', '.join(missing)}")

    token = _get_token(args.tenant_id, args.client_id, args.client_secret)
    app = _request(
        "GET",
        f"{BASE_URL}/my/applications/{args.product_id}",
        token,
    )
    submission_id = _pending_submission_id(app)
    if not submission_id:
        print("No pending submission to commit.")
        return 0

    if args.verbose:
        print(f"Committing pending submission {submission_id}...")
    commit_url = (
        f"{BASE_URL}/my/applications/{args.product_id}/submissions/"
        f"{submission_id}/commit"
    )
    commit_response = _request("POST", commit_url, token)
    commit_status = _lookup(commit_response, "status", "Status")
    if args.verbose:
        print(f"Commit accepted with status {commit_status!r}")

    final_status = _poll_commit(
        token,
        args.product_id,
        submission_id,
        timeout_s=args.timeout,
        interval_s=args.interval,
        verbose=args.verbose,
    )
    status = _lookup(final_status, "status", "Status")
    print(f"Partner Center submission commit complete (status={status}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
