import os, io, base64, logging, traceback, re
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# ─── optional head-less chart libs ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Groq / OpenAI-compatible client ───────────────────────────────────────────
from openai import OpenAI                        # requires openai>=1.14

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ─── FastAPI + CORS for Lovable front-end ──────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or restrict to ["https://your-lovable-url.com"]
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

logger = logging.getLogger("uvicorn.error")

# ─── load the client CSV once at start-up ───────────────────────────────────────
df = pd.read_csv("mock_wealth_clients_with_names.csv")

class Query(BaseModel):
    question: str

# ─── helper: numeric text → plain number string ────────────────────────────────
def _to_number(txt: str) -> str:
    txt = txt.lower().replace("$", "").replace(",", "").strip()
    if txt.endswith("k"):
        return str(float(txt[:-1]) * 1_000)
    if txt.endswith("m"):
        return str(float(txt[:-1]) * 1_000_000)
    return txt

# ─── helper: natural text condition → pandas query string ──────────────────────
_field_map = {
    "cash savings":  "Cash_Savings",
    "cash":          "Cash_Savings",
    "savings":       "Cash_Savings",
    "annual income": "Annual_Income",
    "income":        "Annual_Income",
    "super balance": "Super_Balance",
    "super":         "Super_Balance",
}
_op_map = {
    "greater than": ">",
    "more than":    ">",
    ">":            ">",
    "less than":    "<",
    "<":            "<",
    "equals":       "==",
    "=":            "==",
}
def normalise_condition(question: str) -> str:
    q = question.lower()
    # locate metric
    field = next((col for hint, col in _field_map.items() if hint in q), None)
    if not field:
        raise ValueError("metric not recognised")

    m = re.search(r"(greater than|more than|less than|equals|[><=])\s+\$?([\d,\.]+[kKmM]?)", q)
    if not m:
        raise ValueError("value/comparator not recognised")
    op    = _op_map[m.group(1)]
    value = _to_number(m.group(2))
    return f"{field} {op} {value}"

# ─── helper: tiny pie chart → base64 string ─────────────────────────────────────
def create_pie_chart(row: pd.Series) -> str:
    labels = ["Cash Savings", "Super Balance"]
    sizes  = [row["Cash_Savings"], row["Super_Balance"]]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%")
    ax.set_title(f"{row['Name']} – asset mix")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ─── core QA dispatcher ────────────────────────────────────────────────────────
def handle_question(q: str) -> tuple[str, str]:
    q_lower = q.lower()

    # 1️⃣ direct single-client look-ups
    name_hits = [n for n in df["Name"] if n.lower() in q_lower]
    if name_hits:
        client = df[df["Name"] == name_hits[0]].iloc[0]
        if "income" in q_lower:
            return f"{client['Name']}'s annual income is ${client['Annual_Income']:,}.", ""
        if "super" in q_lower:
            return f"{client['Name']}'s super balance is ${client['Super_Balance']:,}.", ""
        if "cash" in q_lower or "savings" in q_lower:
            return f"{client['Name']} has ${client['Cash_Savings']:,} in cash savings.", ""
        if "chart" in q_lower:
            return "Here’s the asset split chart.", create_pie_chart(client)

    # 2️⃣ filtered list / aggregate
    try:
        query_string = normalise_condition(q_lower)
        filtered_df  = df.query(query_string)

        if filtered_df.empty:
            return "No clients match that filter.", ""

        if "how many" in q_lower or "count" in q_lower:
            return f"{len(filtered_df)} clients match that condition.", ""

        names = "\n".join(f"- {n}" for n in filtered_df["Name"])
        return names, ""
    except Exception as e:
        # log + fallback to LLM
        logger.info("tabular parse miss: %s", e)

    # 3️⃣ fallback: Groq Llama-3
    if not OPENAI_API_KEY:
        return "Language model not configured on server.", ""
    try:
        chat = llm.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system",
                 "content": "You are a helpful financial assistant with access to some client data."},
                {"role": "user", "content": q},
            ],
            max_tokens=256,
        )
        return chat.choices[0].message.content.strip(), ""
    except Exception:
        logger.error("Groq LLM call failed:\n%s", traceback.format_exc())
        return "Sorry, I couldn’t reach the language model right now.", ""

# ─── FastAPI route ─────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_ai(query: Query):
    try:
        answer, chart64 = handle_question(query.question)
        return {"answer": answer, "chart_base64": chart64 or ""}
    except Exception:
        logger.error("Unhandled error\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=200,
            content={"answer": "Sorry, I hit an internal error.", "chart_base64": ""}
        )
