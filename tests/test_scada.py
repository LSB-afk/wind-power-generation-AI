from dataclasses import FrozenInstanceError, replace
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import scada as scada_module
from scada import (
    WeightCalibration,
    build_weighted_targets,
    estimate_turbine_weights,
    reconstruct_weighted_potential,
)


def _scada_frame(
    index: pd.DatetimeIndex,
    powers: list[list[float]],
    winds: list[list[float]],
    *,
    maker: str = "test",
) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for position in range(len(powers[0])):
        turbine = position + 1
        data[f"{maker}_wtg{turbine:02d}_power_kw10m"] = [
            row[position] for row in powers
        ]
        data[f"{maker}_wtg{turbine:02d}_ws"] = [row[position] for row in winds]
    return pd.DataFrame(data, index=index)


def _calibration(
    weights: tuple[float, ...], *, maker: str = "test", group: str = "synthetic"
) -> WeightCalibration:
    columns = tuple(
        f"{maker}_wtg{position:02d}_power_kw10m"
        for position in range(1, len(weights) + 1)
    )
    return WeightCalibration(
        group=group,
        train_label_years=(2022,),
        turbine_columns=columns,
        weights=weights,
        calibration_row_count=2,
        calibration_index_hash="index-sha",
        weights_hash=scada_module._hash_weights(columns, weights),
    )


def test_weight_calibration_is_immutable_and_uses_label_year_boundary():
    index = pd.DatetimeIndex(
        [
            "2023-01-01 00:00:00",
            "2022-12-31 23:50:00",
            "2022-12-31 23:00:00",
        ]
    )
    frame = _scada_frame(
        index,
        powers=[[10.0, 20.0, 30.0]] * 3,
        winds=[[6.0, 6.0, 6.0]] * 3,
    )

    calibration = estimate_turbine_weights(
        frame,
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        train_label_years=(2022,),
    )

    assert calibration.train_label_years == (2022,)
    assert calibration.calibration_row_count == 1
    assert calibration.weights == pytest.approx((0.5, 1.0, 1.5))
    with pytest.raises(FrozenInstanceError):
        calibration.weights = (1.0, 1.0, 1.0)


def test_weight_estimation_is_deterministic_for_unsorted_rows_and_columns():
    index = pd.date_range("2022-01-01 00:10", periods=3, freq="10min")
    frame = _scada_frame(
        index,
        powers=[[10.0, 20.0, 30.0], [20.0, 40.0, 60.0], [30.0, 60.0, 90.0]],
        winds=[[6.0, 7.0, 8.0]] * 3,
    )
    ordered = estimate_turbine_weights(
        frame,
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        train_label_years=(2022,),
    )
    shuffled = estimate_turbine_weights(
        frame.iloc[::-1, ::-1],
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        train_label_years=(2022,),
    )

    assert shuffled == ordered
    assert np.mean(ordered.weights) == pytest.approx(1.0)
    assert ordered.turbine_columns == tuple(
        f"test_wtg{turbine:02d}_power_kw10m" for turbine in (1, 2, 3)
    )


def test_weight_estimation_rejects_partial_sentinel_and_low_output_rows():
    index = pd.date_range("2022-01-01 00:10", periods=5, freq="10min")
    frame = _scada_frame(
        index,
        powers=[
            [10.0, 20.0, 30.0],
            [10.0, np.nan, 30.0],
            [10.0, 20.0, 50_000_000.0],
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
        winds=[
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 4.9, 6.0],
        ],
    )

    calibration = estimate_turbine_weights(
        frame,
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        train_label_years=(2022,),
    )

    assert calibration.calibration_row_count == 1
    assert calibration.weights == pytest.approx((0.5, 1.0, 1.5))


def test_equal_weights_match_v5_and_all_healthy_rows_are_identity():
    index = pd.DatetimeIndex(["2022-01-01 00:10", "2022-01-01 00:20"])
    frame = _scada_frame(
        index,
        powers=[[10.0, 20.0, 0.0], [10.0, 20.0, 30.0]],
        winds=[[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
    )

    result = reconstruct_weighted_potential(
        frame,
        _calibration((1.0, 1.0, 1.0)),
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        min_healthy=2,
    )

    assert result.iloc[0] == pytest.approx((10.0 + 20.0) / 2.0 * 3.0)
    assert result.iloc[1] == pytest.approx(10.0 + 20.0 + 30.0)


def test_weighted_coverage_handles_missing_high_and_low_weight_turbines():
    index = pd.DatetimeIndex(["2022-01-01 00:10", "2022-01-01 00:20"])
    frame = _scada_frame(
        index,
        powers=[[5.0, 10.0, 0.0], [0.0, 10.0, 15.0]],
        winds=[[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
    )

    result = reconstruct_weighted_potential(
        frame,
        _calibration((0.5, 1.0, 1.5)),
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        min_healthy=2,
    )

    np.testing.assert_allclose(result.to_numpy(), [30.0, 30.0])


def test_reconstruction_rejects_weights_that_do_not_match_their_hash():
    frame = _scada_frame(
        pd.DatetimeIndex(["2022-01-01 00:10"]),
        powers=[[10.0, 20.0, 30.0]],
        winds=[[6.0, 6.0, 6.0]],
    )
    calibration = replace(
        _calibration((0.5, 1.0, 1.5)), weights_hash="stale-weights-sha"
    )

    with pytest.raises(ValueError, match="weights hash"):
        reconstruct_weighted_potential(
            frame,
            calibration,
            group="synthetic",
            maker="test",
            turbines=(1, 2, 3),
            capacity_kwh=600.0,
            min_healthy=2,
        )


def test_reconstruction_sanitizes_values_clips_each_interval_and_stays_finite():
    index = pd.date_range("2022-01-01 00:10", periods=3, freq="10min")
    frame = _scada_frame(
        index,
        powers=[
            [500.0, 500.0, 500.0],
            [50_000_000.0, 20.0, 30.0],
            [np.nan, 0.0, 30.0],
        ],
        winds=[[6.0, 6.0, 6.0]] * 3,
    )

    result = reconstruct_weighted_potential(
        frame,
        _calibration((0.5, 1.0, 1.5)),
        group="synthetic",
        maker="test",
        turbines=(1, 2, 3),
        capacity_kwh=600.0,
        min_healthy=2,
    )

    assert result.iloc[0] == 100.0
    assert result.iloc[1] == pytest.approx(50.0 * 3.0 / 2.5)
    assert np.isnan(result.iloc[2])
    assert not np.isinf(result.to_numpy()).any()


@pytest.mark.parametrize(
    ("n_turbines", "min_healthy", "expected_valid"),
    [(6, 3, True), (6, 3, False), (5, 2, True), (5, 2, False)],
)
def test_reconstruction_pins_group_minimum_healthy_counts(
    n_turbines: int, min_healthy: int, expected_valid: bool
):
    healthy = min_healthy if expected_valid else min_healthy - 1
    powers = [10.0] * healthy + [0.0] * (n_turbines - healthy)
    frame = _scada_frame(
        pd.DatetimeIndex(["2022-01-01 00:10"]),
        powers=[powers],
        winds=[[6.0] * n_turbines],
    )

    result = reconstruct_weighted_potential(
        frame,
        _calibration((1.0,) * n_turbines),
        group="synthetic",
        maker="test",
        turbines=tuple(range(1, n_turbines + 1)),
        capacity_kwh=600.0,
        min_healthy=min_healthy,
    )

    assert bool(result.notna().iloc[0]) is expected_valid


def test_build_weighted_targets_accepts_only_indexes_and_masks_other_label_years():
    index = pd.DatetimeIndex(["2022-01-01 01:00", "2023-01-01 01:00"])
    powers = [[100.0] * 6, [120.0] * 6]
    frame = _scada_frame(index, powers=powers, winds=[[6.0] * 6] * 2, maker="vestas")

    targets, calibrations = build_weighted_targets(
        index,
        train_label_years=(2022,),
        target_label_years=(2022,),
        groups=("kpx_group_1",),
        scada_frames={"vestas": frame},
    )

    assert targets.loc[index[0], "kpx_group_1_weighted_potential"] == 3600.0
    assert np.isnan(targets.loc[index[1], "kpx_group_1_weighted_potential"])
    assert tuple(calibrations) == ("kpx_group_1",)
    assert calibrations["kpx_group_1"].train_label_years == (2022,)
    assert calibrations["kpx_group_1"].group == "kpx_group_1"


def test_build_weighted_targets_never_calibrates_unrequested_group3():
    index = pd.DatetimeIndex(["2022-01-01 01:00"])
    vestas = _scada_frame(
        index,
        powers=[[100.0] * 12],
        winds=[[6.0] * 12],
        maker="vestas",
    )

    targets, calibrations = build_weighted_targets(
        index,
        train_label_years=(2022,),
        groups=("kpx_group_1", "kpx_group_2"),
        scada_frames={"vestas": vestas},
    )

    assert tuple(calibrations) == ("kpx_group_1", "kpx_group_2")
    assert not any("group_3" in column for column in targets.columns)


def test_weighted_target_builder_cannot_observe_changed_label_values():
    index = pd.DatetimeIndex(["2022-01-01 01:00"])
    frame = _scada_frame(
        index,
        powers=[[100.0] * 6],
        winds=[[6.0] * 6],
        maker="vestas",
    )
    labels_a = pd.DataFrame({"kpx_group_1": [1000.0]}, index=index)
    labels_b = pd.DataFrame({"kpx_group_1": [9999.0]}, index=index)

    targets_a, _ = build_weighted_targets(
        labels_a.index,
        train_label_years=(2022,),
        groups=("kpx_group_1",),
        scada_frames={"vestas": frame},
    )
    targets_b, _ = build_weighted_targets(
        labels_b.index,
        train_label_years=(2022,),
        groups=("kpx_group_1",),
        scada_frames={"vestas": frame},
    )

    pd.testing.assert_frame_equal(targets_a, targets_b)
