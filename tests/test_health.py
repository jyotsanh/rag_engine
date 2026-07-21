from fastapi.testclient import TestClient

from rag_engine.api.app import app

# Constructed WITHOUT a `with` block so the lifespan (and thus a live MongoDB)
# does not run — `/health` is pure liveness and needs no backing services.
client = TestClient(app)


def test_health_ok():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"message": "app is healthy!!!"}
