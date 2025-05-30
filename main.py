# ---------------- standard libs ----------------
import os, io, base64, logging, traceback, subprocess

# ---------------- 3rd-party libs ---------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # head-less backend for Render
import matplotlib.pyplot as plt
import openai                  # works for Groq-compat too

# ================= FastAPI app =================
app = FastAPI()

# --- allow Lovable (or ANY origin) to call the API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to ["https://your-lovable-url.com"] if you wish
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

logger = logging.getLogger("uvicorn.error")

# ================== external LLM ===============
openai.api_key  = os.getenv("OPENAI_API_KEY", "")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

# ================ load client CSV ==============
df = pd.read_csv("mock_wealth_clients_with_names.csv")

# ================ schema =======================
class Query(BaseModel):
    question: str

# ================ helpers ======================
def generate_chart(client_row: pd.Series) -> str | None:
    """
    Build a pie chart of a single client’s asset split (Cash vs Super).
    Returns a base64 PNG or None if we decide no chart is needed.
    """
    labels  = ["Cash Savings", "Super Balance"]
    sizes   = [client_row["Cash_Savings"], client_row["Super_Balance"]]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%")
    ax.set_title(f"{client_row['Name']} – asset mix")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def ask_groq_llm(prompt: str) -> str:
    """
    Send the prompt to Groq’s OpenAI-compatible endpoint (model 'mixtral-8x7b-32768' or similar).
    """
    try:
        completion = openai.ChatCompletion.create(
            model="mixtral-8x7b-32768",
            messages=[{"role":"user","content": prompt}],
            timeout=15,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Groq LLM call failed: %s", e)
        return "Sorry, I couldn’t reach the language model right now."

def handle_question(q: str) -> tuple[str, str | None]:
    """
    Core logic – returns (answer, optional_chart_b64)
    """
    question = q.lower()
    # try to locate a client name mentioned in the question
    names = [name for name in df["Name"] if name.lower() in question]
    if not names:
        # let Groq handle free-form questions that aren’t client-specific
        return ask_groq_llm(q), None

    client = df[df["Name"] == names[0]].iloc[0]

    # ---- hard-coded examples ---------------------------------
    if "income" in question:
        return (
            f"{client['Name']}'s annual income is ${client['Annual_Income']:,}.",
            None,
        )
    if "super" in question:
        return (
            f"{client['Name']}'s super balance is ${client['Super_Balance']:,}.",
            None,
        )
    if "cash" in question or "savings" in question:
        return (
            f"{client['Name']} has ${client['Cash_Savings']:,} in cash savings.",
            None,
        )
    if "asset mix" in question or "chart" in question:
        ch_b64 = generate_chart(client)
        return "Here’s the asset split chart.", ch_b64

    # Fallback → combine CSV facts + Groq reasoning
    sys_prompt = (
        "You are an AI wealth-management assistant. "
        "Use the structured data provided if relevant; otherwise think step-by-step."
    )
    full_prompt = (
        f"{sys_prompt}\n\nStructured data:\n{client.to_json()}\n\nUser question: {q}"
    )
    return ask_groq_llm(full_prompt), None

# ============= FastAPI route ===================
@app.post("/ask")
def ask_ai(query: Query):
    try:
        answer, chart64 = handle_question(query.question)
        return {"answer": answer, "chart_base64": chart64 or ""}
    except Exception:
        logger.error("Unhandled error\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=200,
            content={
                "answer": "Sorry, I hit an internal error.",
                "chart_base64": "",
            },
        )
