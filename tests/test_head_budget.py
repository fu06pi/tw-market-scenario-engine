import json

import pytest

from tw_scenario.quant import head_budget
from tw_scenario.quant.head_budget import HeadBudget


def test_budget_is_cumulative_across_resumes_and_candidates(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(head_budget.time, "monotonic", lambda: clock[0])
    path = tmp_path / "budget.json"
    with HeadBudget(path, "candidate0", 20, 10) as first:
        assert first.allowance == 10
        clock[0] += 7
    with HeadBudget(path, "candidate0", 20, 10) as resumed:
        assert resumed.allowance == 3
        clock[0] += 3
    with (pytest.raises(TimeoutError, match="cumulative"),
          HeadBudget(path, "candidate0", 20, 10)):
        pytest.fail("exhausted run admitted")
    with HeadBudget(path, "candidate1", 20, 10):
        clock[0] += 10
    with pytest.raises(TimeoutError), HeadBudget(path, "candidate2", 20, 10):
        pytest.fail("exhausted search admitted")
    assert sum(a["charged_seconds"] for a in json.loads(path.read_text())["attempts"]) == 20


def test_budget_reservation_survives_unclean_exit_and_refuses_changed_limits(tmp_path):
    path = tmp_path / "budget.json"
    attempt = HeadBudget(path, "run", 20, 10)
    attempt.__enter__()
    # Simulate process loss: no __exit__, OS releases lock, reservation remains.
    attempt.lock.close()
    assert json.loads(path.read_text())["attempts"][0]["charged_seconds"] == 10
    with pytest.raises(TimeoutError), HeadBudget(path, "run", 20, 10):
        pytest.fail("crashed run acquired a fresh budget")
    with pytest.raises(ValueError, match="frozen"), HeadBudget(path, "other", 40, 10):
        pytest.fail("budget silently expanded")


def test_budget_excludes_concurrent_workers_and_charges_failures(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(head_budget.time, "monotonic", lambda: clock[0])
    path = tmp_path / "budget.json"
    with pytest.raises(RuntimeError, match="training failed"), HeadBudget(path, "a", 20, 10):
        with pytest.raises(BlockingIOError), HeadBudget(path, "b", 20, 10):
            pytest.fail("concurrent budget reservation admitted")
        clock[0] += 4
        raise RuntimeError("training failed")
    recorded = json.loads(path.read_text())["attempts"]
    assert len(recorded) == 1
    assert recorded[0]["charged_seconds"] == 4
    assert recorded[0]["status"] == "FAILED"
