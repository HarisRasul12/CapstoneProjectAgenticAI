from __future__ import annotations

import numpy as np
import pandas as pd


def generate_twap_schedule(quantity: int, bars: pd.DataFrame) -> pd.DataFrame:
    weights = pd.Series(np.ones(len(bars), dtype=float), index=bars.index)
    return _schedule_from_weights("TWAP", quantity, bars, weights, "Equal child orders across the window.")


def generate_vwap_schedule(quantity: int, bars: pd.DataFrame, volume_curve: pd.Series) -> pd.DataFrame:
    curve = volume_curve.reindex(bars.index).fillna(0)
    return _schedule_from_weights("VWAP", quantity, bars, curve, "Child orders follow the expected intraday volume curve.")


def generate_pov_schedule(
    quantity: int,
    bars: pd.DataFrame,
    participation_rate: float,
    pov_mode: str = "strict_cap",
) -> pd.DataFrame:
    if bars.empty:
        return _empty_schedule("POV")
    caps = np.floor(bars["volume"].astype(float).clip(lower=0) * participation_rate).astype(int).to_numpy()
    if caps.sum() <= 0:
        weights = pd.Series(np.ones(len(bars)), index=bars.index)
        note = "No market volume was available, so POV fell back to equal slices."
        return _schedule_from_weights("POV", quantity, bars, weights, note)

    strict_child = _allocate_until_complete(quantity, caps)
    strict_unfilled = max(0, int(quantity) - int(strict_child.sum()))
    if pov_mode == "strict_cap":
        child = strict_child
        note = (
            f"Strict POV caps each child order at {participation_rate:.1%} of displayed bar volume. "
            f"Unfilled shares are reported instead of forcing catch-up: {strict_unfilled:,}."
        )
    else:
        child = strict_child.copy()
        if strict_unfilled > 0:
            catch_up = _allocate_integer_quantity(strict_unfilled, bars["volume"].astype(float).clip(lower=0))
            child = child + catch_up.to_numpy(dtype=int)
        note = (
            f"Force-complete POV starts from a {participation_rate:.1%} cap, then allocates catch-up "
            "shares when strict capacity is insufficient. Cap violations are explicitly flagged."
        )

    return _schedule_frame("POV", bars, child, note, parent_quantity=quantity, participation_cap_quantity=caps)


def generate_is_schedule(quantity: int, bars: pd.DataFrame, urgency: float) -> pd.DataFrame:
    if bars.empty:
        return _empty_schedule("IS")
    x = np.linspace(0.0, 1.0, len(bars))
    decay = 1.0 + 5.0 * float(urgency)
    weights = pd.Series(np.exp(-decay * x), index=bars.index)
    note = f"Front-loaded implementation-shortfall style schedule with urgency={urgency:.2f}."
    return _schedule_from_weights("IS", quantity, bars, weights, note)


def _schedule_from_weights(
    algo: str,
    quantity: int,
    bars: pd.DataFrame,
    weights: pd.Series,
    note: str,
) -> pd.DataFrame:
    if bars.empty:
        return _empty_schedule(algo)
    child = _allocate_integer_quantity(quantity, weights.reindex(bars.index).fillna(0))
    return _schedule_frame(algo, bars, child.to_numpy(dtype=int), note, parent_quantity=quantity)


def _schedule_frame(
    algo: str,
    bars: pd.DataFrame,
    child: np.ndarray,
    note: str,
    parent_quantity: int,
    participation_cap_quantity: np.ndarray | None = None,
) -> pd.DataFrame:
    caps = (
        participation_cap_quantity.astype(int)
        if participation_cap_quantity is not None
        else np.full(len(child), -1, dtype=int)
    )
    frame = pd.DataFrame(
        {
            "timestamp_et": list(bars["timestamp_et"]),
            "algo": algo,
            "target_quantity": child.astype(int),
            "parent_quantity": int(parent_quantity),
            "bar_volume": bars["volume"].astype(float).values,
            "participation_cap_quantity": caps,
            "schedule_note": note,
        }
    )
    frame["target_quantity"] = frame["target_quantity"].clip(lower=0).astype(int)
    frame["cap_violation"] = (
        (frame["participation_cap_quantity"] >= 0)
        & (frame["target_quantity"] > frame["participation_cap_quantity"])
    )
    return frame


def _empty_schedule(algo: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp_et", "algo", "target_quantity", "bar_volume", "schedule_note"]
    )


def _allocate_integer_quantity(quantity: int, weights: pd.Series) -> pd.Series:
    clean = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    if len(clean) == 0:
        return clean.astype(int)
    if float(clean.sum()) <= 0:
        clean = pd.Series(np.ones(len(clean)), index=clean.index)
    normalized = clean / float(clean.sum())
    raw = normalized * int(quantity)
    base = np.floor(raw).astype(int)
    remainder = int(quantity) - int(base.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw - base).to_numpy())
        for idx in fractional_order[:remainder]:
            base.iloc[idx] += 1
    elif remainder < 0:
        reduce_order = np.argsort((raw - base).to_numpy())
        for idx in reduce_order[: abs(remainder)]:
            if base.iloc[idx] > 0:
                base.iloc[idx] -= 1
    return base.astype(int)


def _allocate_until_complete(quantity: int, caps: np.ndarray) -> np.ndarray:
    child = np.zeros_like(caps, dtype=int)
    remaining = int(quantity)
    for idx, cap in enumerate(caps.astype(int)):
        if remaining <= 0:
            break
        shares = min(int(cap), remaining)
        child[idx] = shares
        remaining -= shares
    return child


def _repair_integer_sum(child: np.ndarray, quantity: int) -> np.ndarray:
    child = np.clip(child.astype(int), 0, None)
    delta = int(quantity) - int(child.sum())
    idx = 0
    while delta > 0 and len(child) > 0:
        child[idx % len(child)] += 1
        delta -= 1
        idx += 1
    while delta < 0 and len(child) > 0:
        pos = idx % len(child)
        if child[pos] > 0:
            child[pos] -= 1
            delta += 1
        idx += 1
    return child
