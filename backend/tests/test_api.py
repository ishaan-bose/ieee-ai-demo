import json

from app import __version__
from tests.conftest import requires_torch


def test_health_shape(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ok", "device", "gpu_name", "vram_gb", "queue_length", "demo_mode", "paused", "version"}
    assert body["ok"] is True
    assert body["device"] in ("cpu", "cuda")
    if body["device"] == "cpu":
        assert body["gpu_name"] is None and body["vram_gb"] is None
    else:
        assert isinstance(body["gpu_name"], str) and body["vram_gb"] > 0
    assert body["queue_length"] == 0
    assert body["demo_mode"] is False and body["paused"] is False
    assert body["version"] == __version__


def test_health_reflects_db_state(client, app_env):
    from app import db

    conn = db.connect(app_env / "demo.db")
    db.log_admin_event(conn, "pause")
    db.log_admin_event(conn, "demo_mode", {"enabled": True})
    conn.execute("INSERT INTO jobs (kind, status, queue_order, created_at, updated_at) VALUES ('competition','queued',1,0,0)")
    conn.close()
    body = client.get("/api/health").json()
    assert body["paused"] is True and body["demo_mode"] is True and body["queue_length"] == 1


@requires_torch
def test_self_check_ran_and_passed(client):
    sc = client.app.state.self_check
    assert sc is not None and sc.ok, sc.to_dict()
    assert sc.smoke_test["ok"] is True
    assert sc.warnings == []  # fake data is complete


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_hello_stream(client):
    with client.stream("GET", "/api/dev/hello-stream?limit=3&interval=0.01") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        text = "".join(r.iter_text())
    events = parse_sse(text)
    assert [e for e, _ in events] == ["tick", "tick", "tick", "done"]
    assert [d["count"] for _, d in events[:3]] == [0, 1, 2]
    assert all(isinstance(d["server_time"], float) for _, d in events[:3])


def test_error_shape(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.json() == {"error": "not_found", "message": "Not Found"}
    r = client.get("/api/dev/hello-stream?limit=0")
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/health" in paths and "/api/dev/hello-stream" in paths
