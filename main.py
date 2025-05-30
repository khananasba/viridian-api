# ===================== main.py =====================
import os, io, base64, logging, traceback, re, numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import matplotlib, matplotlib.pyplot as plt

matplotlib.use("Agg")  # head-less backend for Render

from openai import OpenAI  # OpenAI-compatible; points to Groq

# ── Groq / OpenAI client ───────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ── FastAPI + CORS ─────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# ── Load client CSV once ───────────────────────────
df = pd.read_csv("mock_wealth_clients_with_names.csv")

class Query(BaseModel):
    question: str

# ╭───────────────────────── Chart helpers ──────────────────────────╮
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def pie_asset_mix(row: pd.Series) -> str:
    labels = ["Cash", "Super"]
    sizes  = [row["Cash_Savings"], row["Super_Balance"]]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=["#1f77b4", "#ff7f0e"])
    ax.set_title(f"{row['Name']} – asset mix")
    return _fig_to_b64(fig)

def bar_compare_two(row_a, row_b) -> str:
    labels = [row_a["Name"], row_b["Name"]]
    width  = 0.25
    x = np.arange(len(labels))
    metrics = [("Cash_Savings", "#4e79a7"),
               ("Super_Balance", "#9c755f"),
               ("Annual_Income", "#f28e2b")]
    fig, ax = plt.subplots(figsize=(6,3))
    for i, (m, c) in enumerate(metrics):
        ax.bar(x+i*width, [row_a[m], row_b[m]], width=width,
               label=m.replace("_"," "), color=c)
    ax.set_xticks(x+width)
    ax.set_xticklabels(labels)
    ax.set_title("Client comparison")
    ax.legend()
    return _fig_to_b64(fig)

def bar_returns_single(row) -> str:
    metrics = [("Cash_Savings", "#4e79a7"),
               ("Super_Balance", "#9c755f"),
               ("Investment_Assets", "#59a14f")]
    returns = np.random.uniform(3, 15, len(metrics))  # mock annual returns %
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar([m[0].replace("_"," ") for m in metrics], returns,
           color=[c for _,c in metrics])
    ax.set_ylabel("Annual return (%)")
    ax.set_title(f"{row['Name']} – mock returns by asset class")
    return _fig_to_b64(fig)

def column_portfolio(row) -> str:
    metrics = [("Cash_Savings","#4e79a7"),
               ("Super_Balance","#9c755f"),
               ("Investment_Assets","#59a14f")]
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar([m[0].replace("_"," ") for m in metrics],
           [row[m[0]] for m in metrics],
           color=[c for _,c in metrics])
    ax.set_ylabel("AUD")
    ax.set_title(f"{row['Name']} – asset columns")
    return _fig_to_b64(fig)

def line_portfolio_growth(row, months=12) -> str:
    today = datetime.today()
    dates = [today - timedelta(days=30*i) for i in reversed(range(months))]
    total_assets = (row["Cash_Savings"]+row["Super_Balance"]+row["Investment_Assets"])
    values = np.linspace(total_assets*0.85, total_assets, months) * np.random.uniform(0.96,1.04,months)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(dates, values, marker="o")
    ax.set_ylabel("AUD")
    ax.set_title(f"{row['Name']} – portfolio performance (simulated)")
    fig.autofmt_xdate()
    return _fig_to_b64(fig)

def scatter_cash_vs_super() -> str:
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(df["Cash_Savings"], df["Super_Balance"], alpha=0.7)
    for _, r in df.iterrows():
        ax.annotate(r["Name"].split()[0], (r["Cash_Savings"], r["Super_Balance"]), fontsize=6)
    ax.set_xlabel("Cash Savings")
    ax.set_ylabel("Super Balance")
    ax.set_title("Cash vs Super across clients")
    return _fig_to_b64(fig)

# ╭───────────────────────── Number-filter helpers ─────────────────╮
def _to_num(txt: str) -> str:
    txt = txt.lower().replace("$","").replace(",","").strip()
    if txt.endswith("k"): return str(float(txt[:-1])*1_000)
    if txt.endswith("m"): return str(float(txt[:-1])*1_000_000)
    return txt

_field_map = {
    "cash savings":"Cash_Savings","cash":"Cash_Savings","savings":"Cash_Savings",
    "annual income":"Annual_Income","income":"Annual_Income",
    "super balance":"Super_Balance","super":"Super_Balance",
}
_op_map = {"greater than":">","more than":">",">":">",
           "less than":"<","<":"<","equals":"==","=":"=="}

def to_query(q: str) -> str:
    field = next((col for k,col in _field_map.items() if k in q), None)
    m = re.search(r"(greater than|more than|less than|equals|[><=])\s+\$?([\d,\.]+[kKmM]?)", q)
    if not field or not m: raise ValueError
    op = _op_map[m.group(1)]; val = _to_num(m.group(2))
    return f"{field} {op} {val}"

# ╭───────────────────────── Core dispatcher ───────────────────────╮
def handle_question(q: str) -> tuple[str,str]:
    ql = q.lower()
    hits = [n for n in df["Name"] if n.lower() in ql]

    # A) single-client quick answers / charts
    if hits:
        row = df[df["Name"] == hits[0]].iloc[0]
        if "income" in ql: return f"{row['Name']}'s annual income is ${row['Annual_Income']:,}.", ""
        if "super"  in ql and "trend" not in ql: return f"{row['Name']}'s super balance is ${row['Super_Balance']:,}.", ""
        if "cash" in ql and "savings" in ql: return f"{row['Name']} has ${row['Cash_Savings']:,} in cash savings.", ""
        if "pie chart" in ql or "asset mix" in ql: return "Pie chart of asset mix.", pie_asset_mix(row)
        if "column chart" in ql or "asset columns" in ql: return "Column chart of asset breakdown.", column_portfolio(row)
        if "bar chart" in ql and "return" in ql: return "Bar chart of asset-class returns.", bar_returns_single(row)
        if "line chart" in ql or "portfolio performance" in ql or "super trend" in ql: return "Line chart of portfolio performance.", line_portfolio_growth(row)

    # B) two-client comparison bar chart
    if "compare" in ql and len(hits) == 2 and "chart" in ql:
        a,b = (df[df["Name"]==n].iloc[0] for n in hits)
        return "Comparison bar chart.", bar_compare_two(a,b)

    # C) scatter plot across all clients
    if "scatter" in ql or "relationship" in ql:
        return "Scatter plot of cash vs super.", scatter_cash_vs_super()

    # D) numeric filter list / count
    try:
        query = to_query(ql); sub = df.query(query)
        if sub.empty: return "No clients match that filter.", ""
        if "how many" in ql or "count" in ql: return f"{len(sub)} clients match that condition.", ""
        return "\n".join(f"- {n}" for n in sub["Name"]), ""
    except Exception: pass  # fall through to LLM

    # E) LLM fallback
    try:
        chat = llm.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role":"user","content": q}],
            max_tokens=256,
        )
        return chat.choices[0].message.content.strip(), ""
    except Exception:
        logger.error("LLM fail\n%s", traceback.format_exc())
        return "Sorry, I couldn’t reach the language model right now.", ""

# ╭───────────────────────── FastAPI route ─────────────────────────╮
@app.post("/ask")
def ask_ai(query: Query):
    try:
        ans, img = handle_question(query.question)
        return {"answer": ans, "chart_base64": img}
    except Exception:
        logger.error("Unhandled\n%s", traceback.format_exc())
        return JSONResponse(status_code=200,
            content={"answer":"Sorry, an internal error occurred.","chart_base64":""})
# ===================== end main.py =====================
