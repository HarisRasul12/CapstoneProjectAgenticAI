from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from execlab.config import Settings
from execlab.schemas import ExecutionRequest
from execlab.service import ExecLabService
from tests.conftest import FakeMarketDataClient


ROOT = Path(__file__).resolve().parent


def main() -> None:
    cases = yaml.safe_load((ROOT / "datasets" / "golden_cases.yaml").read_text())["cases"]
    rubric = yaml.safe_load((ROOT / "rubrics" / "rubric.yaml").read_text())
    settings = Settings(
        app_name="execlab-evals",
        vertex_model="gemini-2.5-flash-lite",
        vertex_model_candidates=("gemini-2.5-flash-lite",),
        gcp_project=None,
        gcp_region="us-central1",
        adk_enabled=False,
        require_adk_success=False,
        allow_transient_fallback=True,
        adk_timeout_seconds=300,
        data_provider="yfinance",
        yfinance_timeout_seconds=5,
        historical_curve_lookback_days=3,
        pretrade_lookback_sessions=5,
        beta_lookback_days=63,
        default_interval="5m",
    )

    results: list[dict] = []
    for case in cases:
        service = ExecLabService(
            settings=settings,
            client=FakeMarketDataClient(settings=settings, trend_bps=float(case.get("trend_bps", 0))),
        )
        request = ExecutionRequest(
            ticker=case["ticker"],
            trade_date=date(2026, 4, 20),
            side=case["side"],
            quantity=int(case["quantity"]),
            spread_bps=float(case.get("spread_bps", 2.0)),
            pov_mode=case.get("pov_mode", "strict_cap"),
            limit_price=case.get("limit_price"),
        )
        result = service.run_backtest(request)
        passed, detail = evaluate_case(case, result)
        results.append({"id": case["id"], "passed": passed, "detail": detail})

    print("\n=== ExecLab Golden/Rubric Eval ===")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} - {item['id']} | {item['detail']}")
    pass_rate = sum(item["passed"] for item in results) / max(1, len(results))
    print(f"pass_rate: {pass_rate:.0%}")
    threshold = float(rubric.get("pass_threshold", 0.85))
    if pass_rate < threshold:
        raise SystemExit(1)


def evaluate_case(case: dict, result) -> tuple[bool, str]:
    checks: list[tuple[bool, str]] = []
    memo = result.memo.model_dump()
    for section in case.get("expected_sections", []):
        value = memo.get(section)
        checks.append((bool(value), f"section:{section}"))
    limitation = case.get("required_limitation", "")
    checks.append((limitation.lower() in result.memo.limitation.lower(), "limitation"))
    for algo, schedule in result.schedules.items():
        if not (case.get("require_pov_unfilled") and algo == "POV"):
            checks.append((int(schedule["target_quantity"].sum()) == result.request.quantity, f"{algo}:sum"))
        checks.append((int(schedule["target_quantity"].min()) >= 0, f"{algo}:nonnegative"))
    for algo, sim in result.simulations.items():
        if sim.metrics.total_quantity_executed > 0:
            checks.append((sim.metrics.avg_fill_price > 0, f"{algo}:avg_fill"))
        if not case.get("require_unfilled_any_algo") and not (
            case.get("require_pov_unfilled") and algo == "POV"
        ):
            checks.append((sim.metrics.total_quantity_executed == result.request.quantity, f"{algo}:qty"))
    checks.append((result.pretrade_report.lookback_sessions > 0, "pretrade:sessions"))
    checks.append((result.expected_cost_report.observation_count > 0, "expected_cost:regression"))
    checks.append((result.beta_risk_report.observation_count > 0, "beta_risk:observations"))
    checks.append((bool(result.beta_risk_report.sector_etf), "beta_risk:etf_map"))
    checks.append((bool(result.beta_risk_report.index_comparison), "beta_risk:index_intraday"))
    checks.append((result.peer_report.analyzed_count > 0, "peer_cluster:analyzed"))
    checks.append((bool(result.causal_report.bullets), "causal_tca:bullets"))
    checks.append((bool(result.debate_report.recommended_algo), "debate:recommendation"))
    checks.append((len(result.counterfactual_report.scenarios) >= 5, "counterfactuals:scenarios"))
    checks.append((bool(result.playbook_report.switch_rules), "playbook:switch_rules"))
    checks.append((bool(result.custom_algo_report.components), "custom_algo:components"))
    checks.append((result.custom_algo_report.simulation.metrics.total_quantity_targeted == result.request.quantity, "custom_algo:simulation"))
    checks.append(
        (
            any("Spread uses a high-low proxy" in caveat for caveat in result.expected_cost_report.caveats),
            "spread:non_nbbo_caveat",
        )
    )
    if case.get("require_pov_unfilled"):
        checks.append((result.simulations["POV"].metrics.unfilled_quantity > 0, "POV:unfilled"))
    if case.get("require_unfilled_any_algo"):
        checks.append(
            (
                any(sim.metrics.unfilled_quantity > 0 for sim in result.simulations.values()),
                "limit:unfilled_any_algo",
            )
        )
    passed = all(ok for ok, _ in checks)
    detail = ", ".join(label for _, label in checks)
    return passed, detail


if __name__ == "__main__":
    main()
