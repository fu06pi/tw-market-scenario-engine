from tw_scenario.metrics import multiclass_brier, realized_direction
from tw_scenario.models import Direction


def test_realized_direction():
    assert realized_direction(10) == Direction.UP
    assert realized_direction(-10) == Direction.DOWN
    assert realized_direction(2) == Direction.FLAT


def test_brier_perfect_prediction():
    assert multiclass_brier({"bull": 1.0, "base": 0.0, "bear": 0.0}, "bull") == 0.0
