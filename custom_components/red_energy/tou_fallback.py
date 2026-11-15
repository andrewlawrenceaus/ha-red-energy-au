"""Recover time-of-use periods for half-hours Red Energy labels "ALLDAY".

Since about November 2026, Red Energy's /usage/interval endpoint labels
every half-hour of a Time of Use plan "ALLDAY" in
primaryConsumptionTariffComponent, including days it had already labelled
PEAK/SHOULDER/OFFPEAK. The half-hours are still priced by period, though:
consumptionDollarIncGst is the half-hour's kWh at its period's rate (plus any
controlled load at the CL rate), exactly as before. Without the label, the
period breakdowns and the CL2/TOU inference (cl2_inference.py) have nothing
to work with, so every interval is rejected as "unknown_tariff".

apply_tou_fallback() fills the period back in from what the API still gives:

1. The half-hour's own price. With no controlled load in it, a half-hour
   costs exactly its kWh times one TOU rate, and that pins its period.
2. Otherwise (controlled load blended in, or no consumption to price), the
   period that price-pinned half-hours show at the same time of day on the
   same kind of day (weekday or weekend) in the fetched data, falling back to
   the nearest time of day that has one, then to either kind of day. The
   network's TOU times aren't in the API, so they're learned from the data
   this way rather than configured. Time of day is the interval's own clock
   (Red Energy reports NEM time, a fixed +10:00), so learned times stay
   consistent within a fetch; a fetch spanning a daylight-saving change can
   mix votes for the hour at a period boundary, and the majority wins.

Public holidays need no special handling: a holiday's half-hours are priced
at the weekend rates and pin themselves by price. Only half-hours with
controlled load blended in rely on the learned times, and controlled load
normally runs overnight, in off-peak.

Runs only for plans whose rates resolve at least two of PEAK, OFFPEAK and
SHOULDER (cl2_inference.resolve_rate_roles), so a genuine anytime plan's
ALLDAY half-hours are left alone. Half-hours that already carry a known TOU
label are left as they are, and also teach the learned times.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from .cl2_inference import resolve_rate_roles

TOU_COMPONENTS = ("PEAK", "OFFPEAK", "SHOULDER")

# A half-hour's cost matches a rate when it's within this of kWh x rate.
# Red Energy gives kWh to 3 decimals and the GST-inclusive cost to 4, so a
# pure half-hour lands within about 0.0001 + 0.0005 x rate; TOU rates differ
# by tens of percent, so this can't match two of them.
_COST_TOLERANCE_DOLLARS = 0.0001
_COST_TOLERANCE_PER_KWH = 0.002

_PERIOD_FIELDS = {
    "PEAK": ("peak_import_usage", "peak_export_usage"),
    "OFFPEAK": ("offpeak_import_usage", "offpeak_export_usage"),
    "SHOULDER": ("shoulder_import_usage", "shoulder_export_usage"),
}


def _slot(interval_start: Any) -> tuple[str, int] | None:
    """Return (day kind, minutes after midnight) for an interval start."""
    if not isinstance(interval_start, str):
        return None
    try:
        start = datetime.fromisoformat(interval_start)
    except ValueError:
        return None
    kind = "weekend" if start.weekday() >= 5 else "weekday"
    return kind, start.hour * 60 + start.minute


def _usable_price(interval: dict[str, Any]) -> tuple[float, float] | None:
    """Return (kWh, GST-inclusive cost) when the half-hour can be priced."""
    if not interval.get("pricing_available") or not interval.get("pricing_reliable"):
        return None
    cost = interval.get("consumption_dollar_incl_gst")
    if cost is None:
        return None
    energy = float(interval.get("consumption_kwh") or 0.0)
    if energy <= 0:
        return None
    return energy, float(cost)


def match_price(interval: dict[str, Any], tou_rates: dict[str, float]) -> str | None:
    """Return the one TOU period whose rate explains the half-hour's cost.

    None when no rate matches (controlled load blended in, or no
    consumption) or, defensively, when more than one does.
    """
    priced = _usable_price(interval)
    if priced is None:
        return None
    energy, cost = priced
    tolerance = _COST_TOLERANCE_DOLLARS + _COST_TOLERANCE_PER_KWH * energy
    matches = [
        component for component, rate in tou_rates.items()
        if abs(energy * rate - cost) <= tolerance
    ]
    return matches[0] if len(matches) == 1 else None


def _learned_period(
    votes: dict[tuple[str, int], Counter[str]], slot: tuple[str, int]
) -> str | None:
    """Return the majority period for a slot, or the nearest slot that has one.

    Looks at the same kind of day first, then at either kind. Ties between
    equally near slots go to the earlier one.
    """
    kind, minute = slot
    for kinds in ((kind,), ("weekday", "weekend")):
        best: tuple[int, int, str] | None = None
        for (vote_kind, vote_minute), counter in votes.items():
            if vote_kind not in kinds or not counter:
                continue
            distance = abs(vote_minute - minute)
            distance = min(distance, 1440 - distance)
            period = counter.most_common(1)[0][0]
            candidate = (distance, vote_minute, period)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None:
            return best[2]
    return None


def _plausible(period: str, interval: dict[str, Any], tou_rates: dict[str, float]) -> str:
    """Check a learned period against the half-hour's price.

    A half-hour can't average more than its period's rate (controlled load
    only lowers the average), so if the learned period's rate is below the
    average price, take the cheapest rate that isn't.
    """
    priced = _usable_price(interval)
    if priced is None:
        return period
    energy, cost = priced
    average = cost / energy
    slack = (_COST_TOLERANCE_DOLLARS + _COST_TOLERANCE_PER_KWH * energy) / energy
    if tou_rates[period] + slack >= average:
        return period
    above = [c for c, rate in tou_rates.items() if rate + slack >= average]
    if not above:
        return period
    return min(above, key=lambda c: tou_rates[c])


def apply_tou_fallback(validated_usage: dict[str, Any], rates: list[dict[str, Any]]) -> dict[str, int] | None:
    """Fill in TOU periods for unlabelled half-hours, in place.

    validated_usage is data_validation.validate_usage_data()'s result; rates
    is the service's validated plan rates. For each half-hour without a known
    TOU label, sets tariff_component to the recovered period and
    tariff_component_source to "price" or "schedule" (keeping the API's
    label in reported_tariff_component), then rebuilds each changed day's
    peak/offpeak/shoulder import and export breakdowns.

    Returns counts of half-hours by source ("reported", "price", "schedule",
    "unresolved"), or None when nothing needed filling in or the plan isn't
    a TOU plan.
    """
    entries = validated_usage.get("usage_data") if isinstance(validated_usage, dict) else None
    if not isinstance(entries, list):
        return None
    days = [entry for entry in entries if isinstance(entry, dict)]
    intervals = [i for entry in days for i in entry.get("intervals") or [] if isinstance(i, dict)]
    if all(i.get("tariff_component") in TOU_COMPONENTS for i in intervals):
        return None

    tou_rates: dict[str, float] = resolve_rate_roles(rates or [])["rates_incl_gst"]
    if len(tou_rates) < 2:
        return None

    counts = {"reported": 0, "price": 0, "schedule": 0, "unresolved": 0}
    votes, pending = _price_pass(intervals, tou_rates, counts)
    _schedule_pass(pending, votes, tou_rates, counts)
    for entry in days:
        _rebuild_breakdowns(entry)
    return counts


def _price_pass(
    intervals: list[dict[str, Any]], tou_rates: dict[str, float], counts: dict[str, int]
) -> tuple[dict[tuple[str, int], Counter[str]], list[dict[str, Any]]]:
    """Label what the API or the price can, and learn the TOU times from it.

    Returns the period votes per (day kind, time of day) and the half-hours
    still without a period.
    """
    votes: dict[tuple[str, int], Counter[str]] = {}
    pending: list[dict[str, Any]] = []
    for interval in intervals:
        component = interval.get("tariff_component")
        if component in TOU_COMPONENTS:
            counts["reported"] += 1
        else:
            component = match_price(interval, tou_rates)
            if component is None:
                pending.append(interval)
                continue
            _set_component(interval, component, "price")
            counts["price"] += 1
        slot = _slot(interval.get("interval_start"))
        if slot is not None:
            votes.setdefault(slot, Counter())[component] += 1
    return votes, pending


def _schedule_pass(
    pending: list[dict[str, Any]],
    votes: dict[tuple[str, int], Counter[str]],
    tou_rates: dict[str, float],
    counts: dict[str, int],
) -> None:
    """Give the remaining half-hours the period learned for their time."""
    for interval in pending:
        slot = _slot(interval.get("interval_start"))
        period = _learned_period(votes, slot) if slot is not None else None
        if period is None or period not in tou_rates:
            counts["unresolved"] += 1
            continue
        _set_component(interval, _plausible(period, interval, tou_rates), "schedule")
        counts["schedule"] += 1


def _set_component(interval: dict[str, Any], component: str, source: str) -> None:
    interval.setdefault("reported_tariff_component", interval.get("tariff_component"))
    interval["tariff_component"] = component
    interval["tariff_component_source"] = source


def _rebuild_breakdowns(entry: dict[str, Any]) -> None:
    """Recompute a day's period breakdowns from its intervals' periods.

    Only for days where this module filled a period in; days labelled by the
    API keep the breakdowns api.py built from the labels.
    """
    intervals = [i for i in entry.get("intervals") or [] if isinstance(i, dict)]
    if not any(i.get("tariff_component_source") for i in intervals):
        return
    totals = {field: 0.0 for fields in _PERIOD_FIELDS.values() for field in fields}
    for interval in intervals:
        fields = _PERIOD_FIELDS.get(interval.get("tariff_component"))
        if fields is None:
            continue
        totals[fields[0]] += float(interval.get("consumption_kwh") or 0.0)
        totals[fields[1]] += float(interval.get("generation_kwh") or 0.0)
    for field, value in totals.items():
        entry[field] = round(value, 3)
