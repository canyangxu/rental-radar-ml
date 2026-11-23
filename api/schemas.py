"""Request and response models for the API. FastAPI turns these into the OpenAPI docs at /docs."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Count = Field(default=0, ge=0, le=1_000_000)


class BuildingFeatures(BaseModel):
    """History of one building before a cutoff date. Every count defaults to 0 and every attribute is optional:
    missing attributes are filled with the training median, exactly as in training. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "borough": "MANHATTAN", "units": 837, "stories": 42, "year_built": 2015, "has_mgmt": True,
        "c12_all": 34, "c12_noise": 13, "c12_mold": 6, "c12_pests": 5, "c3_all": 5, "v12_a": 1, "v12_b": 1,
        "bb24_infested": 1}})

    cutoff: Optional[date] = Field(None, description="Date the history ends; defaults to today. Used for building age.")
    borough: Optional[Literal["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]] = None
    units: Optional[float] = Field(None, ge=1, le=20_000, description="Legal residential units")
    stories: Optional[float] = Field(None, ge=1, le=200)
    year_built: Optional[float] = Field(None, ge=1700, le=2030)
    bldgs_lot: Optional[float] = Field(None, ge=1, le=500, description="Buildings on the tax lot")
    lat: Optional[float] = Field(None, ge=40.4, le=41.0)
    lng: Optional[float] = Field(None, ge=-74.3, le=-73.6)
    has_mgmt: bool = Field(False, description="A management company is registered with HPD")
    # 311 complaints, past 12 and 3 months
    c12_all: int = Count; c12_noise: int = Count; c12_heat: int = Count; c12_street: int = Count
    c12_upkeep: int = Count; c12_mold: int = Count; c12_pests: int = Count; c12_sanitation: int = Count
    c3_all: int = Count; c3_noise: int = Count; c3_heat: int = Count; c3_street: int = Count
    c3_upkeep: int = Count; c3_mold: int = Count; c3_pests: int = Count; c3_sanitation: int = Count
    # HPD violations by class, past 12 and 3 months
    v12_a: int = Count; v12_b: int = Count; v12_c: int = Count; v12_ri: int = Count
    v12_mold: int = Count; v12_pests: int = Count; v12_heat: int = Count
    v3_a: int = Count; v3_b: int = Count; v3_c: int = Count; v3_ri: int = Count
    # bedbug filings, past 24 months
    bb24_infested: int = Count; bb24_reinfested: int = Count


class BatchRequest(BaseModel):
    instances: list[BuildingFeatures] = Field(..., min_length=1, max_length=500)


class Driver(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    effect: Literal["raises", "lowers"]
    contribution: float = Field(description="Change in log-odds from this feature (SHAP value for XGBoost)")


class Band(BaseModel):
    low: float
    high: float
    n: int
    mean_predicted: float
    observed_rate: float = Field(description="Share of test-set buildings in this band that did get a B/C violation")


class Prediction(BaseModel):
    probability: float = Field(description="Probability of at least one class B or C violation in the next 6 months")
    label: int = Field(description="1 if probability >= threshold (threshold chosen on validation)")
    risk_level: Literal["low", "medium", "high"]
    percentile: float = Field(description="Share of NYC buildings with a lower predicted probability")
    threshold: float
    base_rate: float = Field(description="Average predicted probability across all NYC buildings today")
    backtest_band: Optional[Band]
    drivers: list[Driver]
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
