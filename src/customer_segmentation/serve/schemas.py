"""Pydantic request/response schemas for the champion API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

FEATURE_FIELDS = (
    "Transaction_Count",
    "Account_Balance",
    "Total_Credit",
    "Total_Withdrawal",
    "Recency_Days",
    "Account_Age_Days",
    "has_loan",
    "has_card",
)


class FeatureRow(BaseModel):
    account_id: int | str | None = None
    Transaction_Count: float
    Account_Balance: float
    Total_Credit: float
    Total_Withdrawal: float
    Recency_Days: float
    Account_Age_Days: float
    has_loan: float = Field(ge=0, le=1)
    has_card: float = Field(ge=0, le=1)


class InferenceRequest(BaseModel):
    """Score one or more accounts (JSON batch)."""

    model_version: str
    as_of_date: str
    customers: list[FeatureRow] = Field(min_length=1)
    include_applications: bool = False


class SingleInferenceRequest(BaseModel):
    """Score a single account."""

    model_version: str
    as_of_date: str
    account_id: int | str | None = None
    Transaction_Count: float
    Account_Balance: float
    Total_Credit: float
    Total_Withdrawal: float
    Recency_Days: float
    Account_Age_Days: float
    has_loan: float = Field(ge=0, le=1)
    has_card: float = Field(ge=0, le=1)
    include_applications: bool = False

    def to_feature_row(self) -> FeatureRow:
        return FeatureRow(
            account_id=self.account_id,
            Transaction_Count=self.Transaction_Count,
            Account_Balance=self.Account_Balance,
            Total_Credit=self.Total_Credit,
            Total_Withdrawal=self.Total_Withdrawal,
            Recency_Days=self.Recency_Days,
            Account_Age_Days=self.Account_Age_Days,
            has_loan=self.has_loan,
            has_card=self.has_card,
        )


class BatchCsvOptions(BaseModel):
    model_version: str
    as_of_date: str
    include_applications: bool = False
    return_csv: bool = True


def score_payload(
    *,
    run_id: str,
    model_version: str,
    as_of_date: str,
    assignments: list[dict[str, Any]],
    monitoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model": run_id,
        "model_version": model_version,
        "as_of_date": as_of_date,
        "n": len(assignments),
        "assignments": assignments,
        "monitoring": monitoring or {},
    }
