from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    vertex_model: str
    vertex_model_candidates: tuple[str, ...]
    gcp_project: str | None
    gcp_region: str
    adk_enabled: bool
    require_adk_success: bool
    allow_transient_fallback: bool
    adk_timeout_seconds: float
    data_provider: str
    yfinance_timeout_seconds: float
    historical_curve_lookback_days: int
    pretrade_lookback_sessions: int
    beta_lookback_days: int
    default_interval: str


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_candidates(primary: str, raw: str | None) -> tuple[str, ...]:
    candidates = [primary]
    if raw:
        candidates.extend(item.strip() for item in raw.split(",") if item.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return tuple(deduped)


def load_settings() -> Settings:
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or _discover_gcloud_project()
        or None
    )
    model = os.getenv("VERTEX_MODEL", "gemini-2.5-flash-lite")
    return Settings(
        app_name=os.getenv("EXECLAB_APP_NAME", "execlab-ai"),
        vertex_model=model,
        vertex_model_candidates=_as_candidates(
            model,
            os.getenv("EXECLAB_VERTEX_MODEL_CANDIDATES", "gemini-2.5-flash,gemini-2.0-flash"),
        ),
        gcp_project=project,
        gcp_region=os.getenv("GOOGLE_CLOUD_LOCATION", os.getenv("EXECLAB_REGION", "us-central1")),
        adk_enabled=_as_bool(os.getenv("EXECLAB_ADK_ENABLED", "true"), default=True),
        require_adk_success=_as_bool(
            os.getenv("EXECLAB_REQUIRE_ADK_SUCCESS", "true"),
            default=True,
        ),
        allow_transient_fallback=_as_bool(
            os.getenv("EXECLAB_ALLOW_TRANSIENT_FALLBACK", "true"),
            default=True,
        ),
        adk_timeout_seconds=float(os.getenv("EXECLAB_ADK_TIMEOUT_SECONDS", "300")),
        data_provider=os.getenv("EXECLAB_DATA_PROVIDER", "yfinance").strip().lower(),
        yfinance_timeout_seconds=float(os.getenv("EXECLAB_YFINANCE_TIMEOUT_SECONDS", "20")),
        historical_curve_lookback_days=int(os.getenv("EXECLAB_CURVE_LOOKBACK_DAYS", "21")),
        pretrade_lookback_sessions=int(os.getenv("EXECLAB_PRETRADE_LOOKBACK_SESSIONS", "21")),
        beta_lookback_days=int(os.getenv("EXECLAB_BETA_LOOKBACK_DAYS", "126")),
        default_interval=os.getenv("EXECLAB_DEFAULT_INTERVAL", "5m"),
    )


def _discover_gcloud_project() -> str | None:
    candidates: list[str] = []
    which_gcloud = shutil.which("gcloud")
    if which_gcloud:
        candidates.append(which_gcloud)

    local_sdk = Path.home() / "Desktop" / "google-cloud-sdk" / "bin" / "gcloud"
    if local_sdk.exists():
        candidates.append(str(local_sdk))

    for executable in candidates:
        try:
            output = subprocess.check_output(
                [executable, "config", "get-value", "project"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            ).strip()
        except Exception:
            continue
        if output and output != "(unset)":
            return output
    return None
