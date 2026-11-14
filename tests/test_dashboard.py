"""Tests for dashboard.py. Run with `python -m pytest` from the repository root."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import dashboard


def test_every_experiment_is_listed_with_a_title():
    entries = dashboard.experiments()
    assert entries[0]["id"] == dashboard.TESTS
    numbers = [e["number"] for e in entries[1:]]
    assert numbers == sorted(numbers) and len(numbers) >= 13
    assert all(e["title"].startswith("Experiment") for e in entries[1:])


def test_compare_reports_identical_differing_and_new_files(tmp_path, monkeypatch):
    committed = tmp_path / "committed"
    run = tmp_path / "run"
    committed.mkdir()
    run.mkdir()
    monkeypatch.setattr(dashboard, "COMMITTED", str(committed))
    (committed / "a.csv").write_text("task,exact,train_seconds\nx,1.0,0.5\ny,0.25,0.5\n")
    (run / "a.csv").write_text("task,exact,train_seconds\nx,1,0.9\ny,0.25,0.1\n")      # only timing differs
    (committed / "b.csv").write_text("task,exact\nx,1.0\ny,0.5\n")
    (run / "b.csv").write_text("task,exact\nx,1.0\ny,0.75\n")
    (committed / "c.csv").write_text("task,exact\nx,1.0\n")
    (run / "c.csv").write_text("task,exact\nx,1.0\ny,0.5\n")
    (run / "d.csv").write_text("task,exact\nx,1.0\n")
    result = {c["file"]: c for c in dashboard.compare(str(run))}
    assert result["a.csv"]["status"] == "identical"
    assert result["b.csv"]["status"] == "differs" and result["b.csv"]["differences"] == {"exact": 1}
    assert result["c.csv"]["status"] == "differs" and "rows" in result["c.csv"]["note"]
    assert result["d.csv"]["status"] == "new"
    meta = dict(id="r", exp="exp0", state="ok", exit_code=0, seconds=75.0, versions=dashboard.versions())
    text = dashboard.report(meta, list(result.values()))
    assert "1 min 15 s" in text and "a.csv: 2 rows, identical" in text and "DIFFERS in exact (1 cells)" in text


def test_server_answers_locally_and_refuses_everything_else():
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def status(path, data=None, headers=None):
        request = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, body = status("/api/state")
        assert code == 200 and json.loads(body)["running"] is None
        assert status("/")[0] == 200
        assert status("/api/state", headers={"Host": "example.com"})[0] == 403        # not addressed to this computer
        assert status("/api/start", data=b'{"exp": "tests"}')[0] == 403                # no dashboard header
        header = {"X-DNN-Dashboard": "1"}
        assert status("/api/start", data=b'{"exp": "../dnn"}', headers=header)[0] == 400
        assert status("/api/csv?run=..&name=x.csv")[0] == 400
        assert status("/api/csv?run=r&name=../dnn.py")[0] == 400
    finally:
        server.shutdown()
