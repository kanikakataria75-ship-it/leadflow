"""Single-user password gate.

This is not a multi-user product, so there are no accounts, no registration and
no password reset. There is one password, held in an environment variable, and
a signed session cookie proving you knew it.

Design notes that matter for the deploy:

* **Signed, not encrypted.** The cookie carries an expiry and an HMAC over it.
  Nothing secret is inside it, so there is nothing to decrypt -- the only claim
  it makes is "this session was issued by a server holding SESSION_SECRET, and
  it has not expired".
* **HttpOnly + SameSite=Lax.** JavaScript cannot read it, which takes session
  theft off the XSS menu. Lax works because the browser talks to one origin:
  the recommended deploy puts Vercel in front and rewrites ``/api/*`` to the
  backend, so the cookie is first-party. If you instead point the frontend at
  the backend's own domain, this must become ``SameSite=None; Secure`` and the
  backend needs the frontend's exact origin in CORS with credentials enabled.
* **Fails closed.** With no ``APP_PASSWORD`` set the gate does not silently
  disable itself -- it refuses every login, and ``/api/health`` reports
  ``auth_configured: false`` so a misconfigured deploy is visible rather than
  wide open.
* **Constant-time comparison**, so the password cannot be recovered a character
  at a time from response timings.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

COOKIE = "lf_session"
#: Seven days. Long enough not to be a nuisance for a tool one person opens
#: daily, short enough that a stolen laptop is not a permanent grant.
MAX_AGE = 7 * 24 * 3600

#: Paths reachable without a session. Deliberately tiny: the login endpoint
#: itself, the health probe the host pings to keep the service alive, and the
#: static assets that render the login screen.
PUBLIC_PREFIXES = ("/api/auth/", "/api/health", "/assets/", "/favicon", "/robots.txt")

router = APIRouter(prefix="/api/auth")


def _secret() -> bytes:
    s = os.getenv("SESSION_SECRET", "")
    if not s:
        # A random per-process secret means sessions do not survive a restart.
        # That is the correct failure: it degrades to "log in again", never to
        # "everyone is authenticated".
        s = base64.urlsafe_b64encode(os.urandom(32)).decode()
        os.environ["SESSION_SECRET"] = s
    return s.encode()


def password_configured() -> bool:
    return bool(os.getenv("APP_PASSWORD", "").strip())


def check_password(candidate: str) -> bool:
    want = os.getenv("APP_PASSWORD", "").strip()
    if not want:
        return False
    return hmac.compare_digest(candidate.encode(), want.encode())


def _sign(payload: dict[str, Any]) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    sig = hmac.new(_secret(), raw, hashlib.sha256).digest()
    return f"{raw.decode()}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"


def _verify(token: str) -> dict[str, Any] | None:
    try:
        raw_s, sig_s = token.split(".", 1)
    except ValueError:
        return None
    raw = raw_s.encode()
    want = hmac.new(_secret(), raw, hashlib.sha256).digest()
    try:
        got = base64.urlsafe_b64decode(sig_s + "=" * (-len(sig_s) % 4))
    except Exception:
        return None
    if not hmac.compare_digest(want, got):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4)))
    except Exception:
        return None
    if float(data.get("exp", 0)) < time.time():
        return None
    return data


def is_authenticated(request: Request) -> bool:
    tok = request.cookies.get(COOKIE)
    return bool(tok and _verify(tok))


def is_public(path: str) -> bool:
    return path.startswith(PUBLIC_PREFIXES)


class LoginBody(BaseModel):
    password: str


@router.post("/login")
def login(body: LoginBody, response: Response) -> JSONResponse:
    if not password_configured():
        raise HTTPException(
            status_code=503,
            detail="APP_PASSWORD is not set on the server; login is disabled.",
        )
    if not check_password(body.password):
        # One message for both "no password set" and "wrong password" would be
        # friendlier and would also tell an attacker which one it was.
        raise HTTPException(status_code=401, detail="Incorrect password.")

    token = _sign({"sub": "owner", "exp": time.time() + MAX_AGE, "iat": time.time()})
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        COOKIE,
        token,
        max_age=MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=os.getenv("COOKIE_SECURE", "1") != "0",
        path="/",
    )
    return resp


@router.post("/logout")
def logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@router.get("/status")
def status(request: Request) -> JSONResponse:
    return JSONResponse({
        "authenticated": is_authenticated(request),
        "auth_configured": password_configured(),
    })


__all__ = ["COOKIE", "is_authenticated", "is_public", "password_configured", "router"]
