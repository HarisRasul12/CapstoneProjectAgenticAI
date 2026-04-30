from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def serve_main() -> None:
    cmd = [
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8000",
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=Path.cwd()))


def test_main() -> None:
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests", "-v"], cwd=Path.cwd()))


def smoke_main() -> None:
    # Uses live data intentionally; choose a recent weekday for yfinance.
    today = datetime.now(ZoneInfo("America/New_York")).date()
    cursor = today - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    command = (
        "from execlab.config import load_settings; "
        "from execlab.service import ExecLabService; "
        "from execlab.schemas import ExecutionRequest; "
        "s=load_settings(); "
        "s=s.__class__(**{**s.__dict__, 'adk_enabled': False, 'require_adk_success': False}); "
        "r=ExecutionRequest(ticker='SPY', trade_date=__import__('datetime').date.fromisoformat('"
        + cursor.isoformat()
        + "'), quantity=1000, interval='5m'); "
        "res=ExecLabService(settings=s).run_backtest(r); "
        "print(res.memo.thesis); print(res.eda.market_vwap); print(res.adk_status)"
    )
    raise SystemExit(subprocess.call([sys.executable, "-c", command], cwd=Path.cwd()))

