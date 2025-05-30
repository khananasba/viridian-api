# main.py  (ViridianIQ API – Groq-powered)

from __future__ import annotations

import os
import re
import io
import base64
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import openai            # ↳ works for Groq because the endpoint is OpenAI-compatible

# ------------------------------------------------------------------------------
#  Groq / OpenAI client setup ---------------------------------------------------
# ------------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # e.g. https://api.groq.com/openai/v1

if not (OPENAI_API_KEY and OPENAI_BASE_URL):
    raise RuntimeError("OPENAI_API_KEY and/or OPENAI_BASE_URL env-vars missing!")

openai.api_key  = OPENAI_API_KEY
openai.base_url = OPENAI_BASE_URL   # IMPORTANT – points the SDK at Groq

LLM_MODEL = "mixtral-8x7b-32768"     # Groq’s best free model; change if you like


# ------------------------------------------------------------------------------
#  FastAPI boiler-plate ---------------------------------------------------------
# ------------------------------------------------------------------------------

app = FastAPI(title="ViridianIQ AI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
#  Data – load once at start ----------------------------------------------------
# ------------------------------------------------------------------------------

CLIENT_DF = pd.read_csv("mock_wealth_clients_with_names.csv")

NUMERIC_COLS = {
    "annual_income": "Annual_Income",
    "super_balance": "Super_Balance",
    "investment_assets": "Investment_Assets",
    "cash_savings": "Cash_Savings",
    "debt": "Debt",
}


# ------------------------------------------------------------------------------
#  Helpers ----------------------------------------------------------------------
# ------------------------------------------------------------------------------

def list_clients_where(col: str, op: str, threshold: float) -> List[str]:
    """Return a python list of client names satisfying a numeric comparison."""
    if col not in NUMERIC_COLS:
        raise KeyError(f"Unknown numeric field '{col}'")

    series = CLIENT_DF[NUMERIC_COLS[col]]

    if   op == ">":  mask = series >  threshold
    elif op == "<":  mask = series <  threshold
    elif op == ">=": mask = series >= threshold
    elif op == "<=": mask = series <= threshold
    elif op == "==": mask = series == threshold
    else:
        raise ValueError(f"Unsupported operator {op}")

    return CLIENT_DF.loc[mask, "Name"].tolist()


def make_pie_for_client(name: str) -> str:
    """Return a base-64 PNG of a client’s asset allocation pie chart."""
    row = CLIENT_DF.loc[CLIENT_DF["Name"] == name]
    if row.empty:
        raise KeyError("Client not found")

    row = row.iloc[0]
    labels = ["Super", "Investments", "Cash"]
    sizes  = [row["Super_Balance"], row["Investment_Assets"], row["Cash_Savings"]]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140)
    ax.set_title(f"{name} – asset allocation")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_groq(prompt: str) -> str:
    """Forward the question to Groq (OpenAI-compatible endpoint)."""
    response = openai.chat.completions.create(
        model   = LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are ViridianIQ's wealth-advisor copilot."},
            {"role": "user",    "content": prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------------------------
#  API schema -------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Ask(BaseModel):
    question: str


class AIResponse(BaseModel):
    answer: str
    chart_base64: str | None = None     # pie-chart when requested


# ------------------------------------------------------------------------------
#  /ask endpoint ----------------------------------------------------------------
# ------------------------------------------------------------------------------

@app.post("/ask", response_model=AIResponse)
def ask_endpoint(payload: Ask):
    q = payload.question.strip()

    # ------------------------------------------------------------------
    #  1) Simple hard-coded look-ups (name + keyword) ------------------
    # ------------------------------------------------------------------
    lowered = q.lower()
    for name in CLIENT_DF["Name"]:
        if name.lower() in lowered:
            row = CLIENT_DF[CLIENT_DF["Name"] == name].iloc[0]

            if "income" in lowered:
                return AIResponse(
                    answer=f"{name}'s annual income is ${row['Annual_Income']:,}."
                )
            if "super" in lowered:
                return AIResponse(
                    answer=f"{name}'s super balance is ${row['Super_Balance']:,}."
                )
            if any(k in lowered for k in ["cash", "savings"]):
                return AIResponse(
                    answer=f"{name} has ${row['Cash_Savings']:,} in cash savings."
                )
            if "show pie" in lowered or "asset chart" in lowered:
                chart_b64 = make_pie_for_client(name)
                return AIResponse(
                    answer="Here is the asset allocation chart.",
                    chart_base64=chart_b64,
                )

    # ------------------------------------------------------------------
    # 2) Aggregate queries like “clients with cash > 100000” ------------
    # ------------------------------------------------------------------
    m = re.search(
        r"(?:clients?|people)\s+with\s+(cash savings|super balance|annual income)"
        r"\s*(>=|>|<=|<|=|==)\s*\$?([\d,\.]+)",
        lowered,
    )
    if m:
        field_text, op, num_txt = m.groups()
        field_key = field_text.replace(" ", "_")
        threshold = float(num_txt.replace(",", ""))
        names = list_clients_where(field_key, op, threshold)
        if not names:
            return AIResponse(answer="No clients meet that criterion.")
        return AIResponse(
            answer=f"{len(names)} clients meet the condition: {', '.join(names)}."
        )

    # ------------------------------------------------------------------
    # 3) Anything else → Groq LLM --------------------------------------
    # ------------------------------------------------------------------
    try:
        answer = ask_groq(q)
        return AIResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}") from exc
