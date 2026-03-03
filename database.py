import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


def _get_cloudflare_base_url() -> str:
    secrets_obj = getattr(st, "secrets", None)
    if secrets_obj is not None:
        cloudflare_section = secrets_obj.get("cloudflare", {})
        url = str(cloudflare_section.get("worker_url", "")).strip()
        if url:
            return url
    return os.environ.get("CLOUDFLARE_WORKER_URL", "").strip()


def _get_cloudflare_api_token() -> str:
    secrets_obj = getattr(st, "secrets", None)
    if secrets_obj is not None:
        cloudflare_section = secrets_obj.get("cloudflare", {})
        token = str(cloudflare_section.get("api_token", "")).strip()
        if token:
            return token
    return os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()


def _build_headers(token: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def save_user_session(user_id: str, chat_history: List[Dict[str, Any]]) -> bool:
    if not user_id:
        return False
    base_url = _get_cloudflare_base_url()
    token = _get_cloudflare_api_token()
    if not base_url or not token:
        return False
    url = f"{base_url.rstrip('/')}/sessions"
    payload = {"user_id": user_id, "chat_history": chat_history}
    try:
        response = requests.post(url, json=payload, headers=_build_headers(token), timeout=10)
        if response.ok:
            return True
    except Exception:
        return False
    return False


def get_user_session(user_id: str) -> Optional[List[Dict[str, Any]]]:
    if not user_id:
        return None
    base_url = _get_cloudflare_base_url()
    token = _get_cloudflare_api_token()
    if not base_url or not token:
        return None
    url = f"{base_url.rstrip('/')}/sessions/{user_id}"
    try:
        response = requests.get(url, headers=_build_headers(token), timeout=10)
        if not response.ok:
            return None
        data = response.json()
        if isinstance(data, dict) and "chat_history" in data:
            history = data.get("chat_history")
        else:
            history = data
        if isinstance(history, list):
            return history
    except Exception:
        return None
    return None


def verify_login(username: str, password: str) -> bool:
    if not username or not password:
        return False
    base_url = _get_cloudflare_base_url()
    token = _get_cloudflare_api_token()
    if not base_url or not token:
        return False
    url = f"{base_url.rstrip('/')}/login"
    payload = {"username": username, "password": password}
    try:
        response = requests.post(url, json=payload, headers=_build_headers(token), timeout=10)
        if not response.ok:
            return False
        data = response.json()
        if isinstance(data, dict) and "authenticated" in data:
            return bool(data.get("authenticated"))
        return bool(data)
    except Exception:
        return False

