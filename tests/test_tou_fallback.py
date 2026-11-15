"""Tests for tou_fallback.apply_tou_fallback(): recovering TOU periods for
half-hours the API labels "ALLDAY" on a Time of Use plan."""
from datetime import date, timedelta

import pytest

from custom_components.red_energy.cl2_inference import infer_cl2_interval
from custom_components.red_energy.tou_fallback import apply_tou_fallback, match_price

RATES = [
    {"rate_code": "P1", "rate_desc": "Peak", "rate_incl_gst_dollars": 0.45760},
    {"rate_code": "S1", "rate_desc": "Shoulder", "rate_incl_gst_dollars": 0.41745},
    {"rate_code": "O1", "rate_desc": "Off-peak", "rate_incl_gst_dollars": 0.32483},
    {"rate_code": "C1", "rate_desc": "CL1", "rate_incl_gst_dollars": 0.18425},
    {"rate_code": "F1", "rate_desc": "Solar", "rate_incl_gst_dollars": 0.04},
]
TOU = {"PEAK": 0.45760, "SHOULDER": 0.41745, "OFFPEAK": 0.32483}
CL_RATE = 0.18425
ANYTIME_RATES = [
    {"rate_code": "A1", "rate_desc": "Anytime", "rate_incl_gst_dollars": 0.35000},
    {"rate_code": "C1", "rate_desc": "CL1", "rate_incl_gst_dollars": 0.18425},
]


def _period(day: date, minute: int, holiday: bool = False) -> str:
    """A typical network schedule: peak 2-8 pm weekdays, off-peak 10 pm-7 am."""
    if minute >= 22 * 60 or minute < 7 * 60:
        return "OFFPEAK"
    if 14 * 60 <= minute < 20 * 60 and day.weekday() < 5 and not holiday:
        return "PEAK"
    return "SHOULDER"


def _day(day: date, *, label: str = "ALLDAY", holiday: bool = False, cl_kwh: float = 0.5,
         general_kwh: float = 0.4, zero_from: int = 10 * 60, zero_to: int = 13 * 60):
    """One day's 48 intervals as api.py normalizes them.

    Controlled load (cl_kwh per half-hour) runs midnight to 4 am; between
    zero_from and zero_to the house runs on solar (no grid import, some
    export). Costs are rounded to 4 decimals, like the API's.
    """
    intervals = []
    for slot in range(48):
        minute = slot * 30
        period = _period(day, minute, holiday)
        general = 0.0 if zero_from <= minute < zero_to else general_kwh
        cl = cl_kwh if minute < 4 * 60 else 0.0
        energy = round(general + cl, 3)
        cost = round(general * TOU[period] + cl * CL_RATE, 4)
        intervals.append({
            "interval_start": f"{day.isoformat()}T{minute // 60:02d}:{minute % 60:02d}:00+10:00",
            "consumption_kwh": energy,
            "generation_kwh": 0.3 if zero_from <= minute < zero_to else 0.0,
            "consumption_dollar_incl_gst": cost,
            "tariff_component": period if label == "TRUE" else label,
            "pricing_available": True,
            "pricing_reliable": True,
            "_true_period": period,
            "_true_cl": cl,
        })
    return {
        "date": day.isoformat(),
        "import_usage": round(sum(i["consumption_kwh"] for i in intervals), 3),
        "peak_import_usage": 0.0,
        "offpeak_import_usage": 0.0,
        "shoulder_import_usage": 0.0,
        "peak_export_usage": 0.0,
        "offpeak_export_usage": 0.0,
        "shoulder_export_usage": 0.0,
        "intervals": intervals,
    }


def _usage(days):
    return {"usage_data": days}


def _week(start=date(2026, 11, 2), n=7, **kwargs):
    """n days from a Monday."""
    return [_day(start + timedelta(days=d), **kwargs) for d in range(n)]


def _all_intervals(usage):
    return [i for e in usage["usage_data"] for i in e["intervals"]]


class TestMatchPrice:
    def test_pure_interval_matches_its_rate(self):
        interval = {"consumption_kwh": 0.312, "consumption_dollar_incl_gst": round(0.312 * TOU["OFFPEAK"], 4),
                    "pricing_available": True, "pricing_reliable": True}
        assert match_price(interval, TOU) == "OFFPEAK"

    def test_small_interval_still_matches(self):
        interval = {"consumption_kwh": 0.015, "consumption_dollar_incl_gst": round(0.015 * TOU["PEAK"], 4),
                    "pricing_available": True, "pricing_reliable": True}
        assert match_price(interval, TOU) == "PEAK"

    def test_blended_interval_matches_nothing(self):
        cost = round(0.4 * TOU["OFFPEAK"] + 0.5 * CL_RATE, 4)
        interval = {"consumption_kwh": 0.9, "consumption_dollar_incl_gst": cost,
                    "pricing_available": True, "pricing_reliable": True}
        assert match_price(interval, TOU) is None

    @pytest.mark.parametrize("override", [
        {"consumption_kwh": 0.0, "consumption_dollar_incl_gst": 0.0},
        {"consumption_dollar_incl_gst": None},
        {"pricing_available": False},
        {"pricing_reliable": False},
    ])
    def test_unpriceable_interval_matches_nothing(self, override):
        interval = {"consumption_kwh": 0.312, "consumption_dollar_incl_gst": round(0.312 * TOU["OFFPEAK"], 4),
                    "pricing_available": True, "pricing_reliable": True, **override}
        assert match_price(interval, TOU) is None


class TestApplyTouFallback:
    def test_recovers_every_period(self):
        usage = _usage(_week())
        counts = apply_tou_fallback(usage, RATES)

        intervals = _all_intervals(usage)
        assert counts["unresolved"] == 0
        assert counts["reported"] == 0
        assert counts["price"] + counts["schedule"] == len(intervals)
        assert all(i["tariff_component"] == i["_true_period"] for i in intervals)
        assert all(i["reported_tariff_component"] == "ALLDAY" for i in intervals)

    def test_blended_and_zero_intervals_come_from_schedule(self):
        usage = _usage(_week())
        apply_tou_fallback(usage, RATES)

        for interval in _all_intervals(usage):
            blended = interval["_true_cl"] > 0
            no_import = interval["consumption_kwh"] == 0
            expected = "schedule" if blended or no_import else "price"
            assert interval["tariff_component_source"] == expected

    def test_cl2_inference_recovers_the_controlled_load(self):
        usage = _usage(_week())
        apply_tou_fallback(usage, RATES)

        results = [infer_cl2_interval(i, TOU, CL_RATE) for i in _all_intervals(usage)]
        assert all(r["accepted"] for r in results)
        true_cl = sum(i["_true_cl"] for i in _all_intervals(usage))
        assert sum(r["cl2_kwh"] for r in results) == pytest.approx(true_cl, abs=0.05)

    def test_rebuilds_breakdowns(self):
        usage = _usage(_week(n=1))
        apply_tou_fallback(usage, RATES)

        entry = usage["usage_data"][0]
        expected = {p: 0.0 for p in TOU}
        expected_export = {p: 0.0 for p in TOU}
        for interval in entry["intervals"]:
            expected[interval["_true_period"]] += interval["consumption_kwh"]
            expected_export[interval["_true_period"]] += interval["generation_kwh"]
        assert entry["peak_import_usage"] == pytest.approx(expected["PEAK"], abs=0.001)
        assert entry["shoulder_import_usage"] == pytest.approx(expected["SHOULDER"], abs=0.001)
        assert entry["offpeak_import_usage"] == pytest.approx(expected["OFFPEAK"], abs=0.001)
        assert entry["shoulder_export_usage"] == pytest.approx(expected_export["SHOULDER"], abs=0.001)
        assert entry["peak_import_usage"] + entry["shoulder_import_usage"] + entry["offpeak_import_usage"] == \
            pytest.approx(entry["import_usage"], abs=0.01)

    def test_weekend_afternoon_learned_separately_from_weekday(self):
        # Weekday 2 pm is peak, weekend 2 pm shoulder. A zero-import Saturday
        # 2 pm must take the weekend's learned period, not the weekdays'.
        days = _week(zero_from=0, zero_to=0)
        days[5] = _day(date(2026, 11, 7), zero_from=14 * 60, zero_to=14 * 60 + 30)
        usage = _usage(days)
        apply_tou_fallback(usage, RATES)

        saturday_2pm = usage["usage_data"][5]["intervals"][28]
        assert saturday_2pm["consumption_kwh"] == 0
        assert saturday_2pm["tariff_component_source"] == "schedule"
        assert saturday_2pm["tariff_component"] == "SHOULDER"
        assert usage["usage_data"][0]["intervals"][28]["tariff_component"] == "PEAK"

    def test_public_holiday_priced_as_weekend_pins_itself(self):
        days = _week()
        days[1] = _day(date(2026, 11, 3), holiday=True)  # a Tuesday holiday
        usage = _usage(days)
        apply_tou_fallback(usage, RATES)

        tuesday_3pm = usage["usage_data"][1]["intervals"][30]
        assert tuesday_3pm["tariff_component"] == "SHOULDER"
        assert tuesday_3pm["tariff_component_source"] == "price"

    def test_nearest_learned_time_fills_a_gap(self):
        # Every day has no import 10 am-1 pm, so those slots have no price
        # votes; they take the nearest learned time (shoulder either side).
        usage = _usage(_week(n=1))
        apply_tou_fallback(usage, RATES)

        eleven = usage["usage_data"][0]["intervals"][22]
        assert eleven["consumption_kwh"] == 0
        assert eleven["tariff_component"] == "SHOULDER"

    def test_learned_period_below_the_price_is_corrected(self):
        # A blended half-hour that averages more than off-peak can't be
        # off-peak, whatever the learned times say.
        usage = _usage(_week(n=1))
        intervals = usage["usage_data"][0]["intervals"]
        odd = intervals[2]  # 1 am, learned as off-peak
        odd["consumption_kwh"] = 0.9
        odd["consumption_dollar_incl_gst"] = round(0.8 * TOU["SHOULDER"] + 0.1 * CL_RATE, 4)
        apply_tou_fallback(usage, RATES)

        assert odd["tariff_component"] == "SHOULDER"
        assert infer_cl2_interval(odd, TOU, CL_RATE)["accepted"]

    def test_labelled_data_is_left_alone(self):
        usage = _usage(_week(label="TRUE"))
        assert apply_tou_fallback(usage, RATES) is None
        assert all("tariff_component_source" not in i for i in _all_intervals(usage))
        assert usage["usage_data"][0]["peak_import_usage"] == 0.0

    def test_labelled_days_teach_unlabelled_ones(self):
        days = _week(label="TRUE", n=6) + [_day(date(2026, 11, 8))]
        usage = _usage(days)
        counts = apply_tou_fallback(usage, RATES)

        assert counts["reported"] == 6 * 48
        assert counts["unresolved"] == 0
        assert all(i["tariff_component"] == i["_true_period"] for i in days[6]["intervals"])
        # Only the unlabelled day's breakdowns are rebuilt.
        assert days[0]["peak_import_usage"] == 0.0
        assert days[6]["offpeak_import_usage"] > 0

    def test_anytime_plan_is_left_alone(self):
        usage = _usage(_week())
        assert apply_tou_fallback(usage, ANYTIME_RATES) is None
        assert all(i["tariff_component"] == "ALLDAY" for i in _all_intervals(usage))

    def test_no_price_votes_leaves_intervals_unresolved(self):
        days = _week(n=1)
        for interval in days[0]["intervals"]:
            interval["pricing_reliable"] = False
        usage = _usage(days)
        counts = apply_tou_fallback(usage, RATES)

        assert counts["unresolved"] == 48
        assert all(i["tariff_component"] == "ALLDAY" for i in days[0]["intervals"])

    @pytest.mark.parametrize("bad", [None, {}, {"usage_data": None}, {"usage_data": [None, {"intervals": None}]}])
    def test_malformed_input(self, bad):
        assert apply_tou_fallback(bad, RATES) is None


class TestCoordinator:
    @pytest.fixture
    def coordinator(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from custom_components.red_energy.coordinator import RedEnergyDataCoordinator

        hass = MagicMock()
        hass.async_add_executor_job = AsyncMock()
        with patch(
            "custom_components.red_energy.coordinator.async_get_clientsession",
            return_value=MagicMock(),
        ):
            coord = RedEnergyDataCoordinator(
                hass=hass,
                username="test_user",
                password="test_pass",
                selected_accounts=["prop-001"],
                services=["electricity"],
            )
        coord.api = AsyncMock()
        return coord

    def _store(self, coordinator, usage, tariff_sources):
        coordinator.data = {
            "usage_data": {
                "prop-001": {
                    "property": {"services": [{"type": "electricity", "rates": RATES}]},
                    "services": {
                        "electricity": {
                            "consumer_number": "elec-123",
                            "usage_data": usage,
                            "tariff_sources": tariff_sources,
                        }
                    },
                }
            }
        }

    def test_cl2_inference_accepts_every_recovered_interval(self, coordinator, caplog):
        usage = _usage(_week())
        service = {"type": "electricity", "consumer_number": "elec-123", "rates": RATES}
        with caplog.at_level("DEBUG", logger="custom_components.red_energy.coordinator"):
            tariff_sources = coordinator._apply_tou_fallback(usage, service)
        self._store(coordinator, usage, tariff_sources)

        result = coordinator.get_cl2_inference("prop-001", "electricity")
        assert result["rejected_interval_count"] == 0
        assert result["accepted_interval_count"] == 7 * 48
        assert result["tariff_sources"] == tariff_sources
        assert result["cl2_energy_kwh"] == pytest.approx(7 * 8 * 0.5, abs=0.05)
        assert result["reconciliation_difference"] == pytest.approx(0.0, abs=0.01)
        assert "Recovered TOU periods for consumer elec-123" in caplog.text
        assert "2026-11-02: periods" in caplog.text

    def test_period_import_usage_uses_recovered_periods(self, coordinator):
        usage = _usage(_week())
        coordinator._apply_tou_fallback(usage, {"consumer_number": "elec-123", "rates": RATES})
        self._store(coordinator, usage, None)

        total = sum(
            coordinator.get_period_import_usage("prop-001", "electricity", period)
            for period in ("PEAK", "SHOULDER", "OFFPEAK")
        )
        assert total == pytest.approx(sum(e["import_usage"] for e in usage["usage_data"]), abs=0.05)
        assert coordinator.get_period_import_usage("prop-001", "electricity", "PEAK") > 0
