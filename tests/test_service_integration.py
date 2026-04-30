from __future__ import annotations

from datetime import date
import importlib

from execlab.config import Settings
from execlab.schemas import ExecutionRequest
from execlab.service import ExecLabService
import execlab.service as service_module
from tests.conftest import FakeMarketDataClient


def test_service_returns_complete_backtest_with_mocked_live_provider() -> None:
    settings = _settings(adk_enabled=False, require_adk_success=False)
    service = ExecLabService(settings=settings, client=FakeMarketDataClient(settings))
    request = ExecutionRequest(ticker="NVDA", trade_date=date(2026, 4, 20), quantity=25_000)

    result = service.run_backtest(request)

    assert result.provider == "fake-live-provider"
    assert set(result.simulations) == {"TWAP", "VWAP", "POV", "IS"}
    assert result.memo.best_algo
    assert "not a production OMS/EMS backtester" in result.memo.limitation
    assert result.eda.market_vwap > 0
    assert result.pretrade_report.lookback_sessions > 0
    assert result.expected_cost_report.observation_count > 0
    assert result.beta_risk_report.observation_count > 0
    assert result.beta_risk_report.sector_etf == "XLK"
    assert result.beta_risk_report.index_comparison
    assert result.peer_report.analyzed_count > 0
    assert result.causal_report.bullets


def test_strict_adk_unavailable_path_returns_labeled_fallback(monkeypatch) -> None:
    settings = _settings(adk_enabled=True, require_adk_success=True)
    monkeypatch.setattr(service_module, "adk_is_available", lambda: False)
    service = ExecLabService(settings=settings, client=FakeMarketDataClient(settings))
    request = ExecutionRequest(ticker="SPY", trade_date=date(2026, 4, 20), quantity=10_000)

    result = service.run_backtest(request)

    assert result.adk_status == "adk_unavailable_fallback"
    assert result.adk_error_summary is not None
    assert result.memo.evidence


def test_streamlit_app_imports() -> None:
    module = importlib.import_module("streamlit_app")
    assert hasattr(module, "main")


def _settings(adk_enabled: bool, require_adk_success: bool) -> Settings:
    return Settings(
        app_name="execlab-test",
        vertex_model="gemini-2.5-flash-lite",
        vertex_model_candidates=("gemini-2.5-flash-lite",),
        gcp_project=None,
        gcp_region="us-central1",
        adk_enabled=adk_enabled,
        require_adk_success=require_adk_success,
        allow_transient_fallback=True,
        data_provider="yfinance",
        yfinance_timeout_seconds=5,
        historical_curve_lookback_days=3,
        pretrade_lookback_sessions=5,
        beta_lookback_days=63,
        default_interval="5m",
    )
