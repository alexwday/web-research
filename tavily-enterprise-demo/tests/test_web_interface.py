import time

from fastapi.testclient import TestClient

from src.adapters.web.app import create_app


def test_web_index_renders():
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "Interactive Use-Case Walkthrough" in response.text


def test_start_internal_check_job_and_poll_completion():
    client = TestClient(create_app())

    start = client.post(
        "/api/use-cases/internal-check/start",
        json={"sample_query": ""},
    )
    assert start.status_code == 200
    payload = start.json()
    assert payload["status"] == "started"
    job_id = payload["job_id"]

    final = None
    for _ in range(80):
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        final = resp.json()
        if final["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert final is not None
    assert final["status"] == "completed"
    assert isinstance(final.get("result"), dict)
    assert "output_path" in final["result"]


def test_unknown_use_case_returns_404():
    client = TestClient(create_app())
    response = client.post("/api/use-cases/unknown-id/start", json={})
    assert response.status_code == 404


def test_index_contains_select_fields():
    """Quarterly-docs should have select dropdowns in the rendered page."""
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "quarterly-docs" in response.text
    assert "All Banks" in response.text


def test_use_cases_endpoint_has_bank_options():
    """The quarterly-docs use case should include bank options."""
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    # The bank options are injected into the JSON blob in the page
    assert "All Banks" in response.text


def test_file_download_nonexistent_returns_404():
    client = TestClient(create_app())
    response = client.get("/api/files/download?path=report/nonexistent.json")
    assert response.status_code == 404


def test_file_view_nonexistent_returns_404():
    client = TestClient(create_app())
    response = client.get("/api/files/view?path=report/nonexistent.json")
    assert response.status_code == 404


def test_file_download_path_traversal_blocked():
    client = TestClient(create_app())
    response = client.get("/api/files/download?path=../../etc/passwd")
    assert response.status_code in (403, 404)
