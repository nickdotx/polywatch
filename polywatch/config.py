"""Configuration: a YAML file plus environment overrides for API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Target(BaseModel):
    address: str
    label: str = ""

    def model_post_init(self, __context) -> None:
        object.__setattr__(self, "address", self.address.lower())


class StrategyConfig(BaseModel):
    """Copy-strategy parameters shared by the backtest and the walk-forward."""

    stake_usd: float = 15.0
    min_copy_usd: float = 100.0       # ignore target trades smaller than this
    entry_min_price: float = 0.05     # skip dust longshots
    entry_max_price: float = 0.95     # skip near-certainties
    categories_allow: list = Field(default_factory=list)
    categories_block: list = Field(default_factory=list)


class CostsConfig(BaseModel):
    slippage_points: float = 0.02
    fee_pct: float = 0.0
    latency_points: float = 0.0


class Endpoints(BaseModel):
    data_api: str = "https://data-api.polymarket.com"
    gamma_api: str = "https://gamma-api.polymarket.com"

    @classmethod
    def from_env(cls) -> "Endpoints":
        return cls(
            data_api=os.getenv("POLYMARKET_DATA_API", "https://data-api.polymarket.com"),
            gamma_api=os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com"),
        )


class Config(BaseModel):
    targets: list[Target] = Field(default_factory=list)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    endpoints: Endpoints = Field(default_factory=Endpoints.from_env)

    @classmethod
    def load(cls, path: str = "config.yaml", env_path: str = ".env") -> "Config":
        _load_dotenv(env_path)
        p = Path(path)
        data = yaml.safe_load(p.read_text()) if p.exists() else {}
        cfg = cls(**(data or {}))
        cfg.endpoints = Endpoints.from_env()
        return cfg


def _load_dotenv(path: str) -> None:
    """Minimal .env loader; does not overwrite existing environment variables."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
