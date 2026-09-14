from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.backend.auth_deps import get_current_user_email

app = FastAPI()


@app.get("/whoami")
def whoami(email: str = Depends(get_current_user_email)):
    return {"email": email}


def test_rejects_when_internal_secret_env_var_unset(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-User-Email": "jane@gmail.com", "X-Internal-Secret": "anything"})
    assert response.status_code == 401


def test_rejects_wrong_secret(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "real-secret")
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-User-Email": "jane@gmail.com", "X-Internal-Secret": "wrong"})
    assert response.status_code == 401


def test_rejects_missing_email_header(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "real-secret")
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-Internal-Secret": "real-secret"})
    assert response.status_code == 401


def test_accepts_correct_secret_and_email(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "real-secret")
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-User-Email": "jane@gmail.com", "X-Internal-Secret": "real-secret"})
    assert response.status_code == 200
    assert response.json() == {"email": "jane@gmail.com"}
