# ============================  main.py  ============================
"""
Viridian AI backend
-------------------------------------------------
• Answers simple “income / super / cash” questions from the CSV
• Generates a bar-chart (base-64 PNG) when a question contains “chart”
• Falls back to Groq’s free Mixtral model for anything else
• Returns: { "answer": "...", "chart_base64": <None or str> }
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os, io, base64, subprocess

# ---------- optional chart libs (head-less) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI()

# ---------- load client data ----------
df = pd.read_csv("mock_wealth_clients_with_names.csv")

class Query(BaseModel):
    question: str


# ---------- helper: make bar chart & return b64 ----------
def build_income_chart(top_n: int = 5) -> str:
    top = df.nlargest(top_n, "Annual_Income")[["Name", "Annual_Income"]]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(top["Name"], top["Annual_Income"], color="#1f77b4")
    ax.set_xlabel("Annual Income ($)")
    ax.set_title(f"Top-{top_n} client incomes")
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------- MAIN ENDPOINT ----------
@app.post("/ask")
def ask_ai(query: Query):
    q = query.question.lower()

    # ----- 1) chart requested? -----
    if "chart" in q:
        chart_b64 = build_income_chart()
        return {
            "answer": "Here’s the chart you asked for:",
            "chart_base64": chart_b64,
        }

    # ----- 2) quick CSV look-ups -----
    name_match = [n for n in df["Name"] if n.lower() in q]
    if name_match:
        client = df.loc[df["Name"] == name_match[0]].iloc[0]

        if "income" in q:
            return {
                "answer": f"{client['Name']}'s annual income is "
                          f"${client['Annual_Income']:,}.",
                "chart_base64": None,
            }
        if "super" in q:
            return {
                "answer": f"{client['Name']}'s super balance is "
                          f"${client['Super_Balance']:,}.",
                "chart_base64": None,
            }
        if "cash" in q or "savings" in q:
            return {
                "answer": f"{client['Name']} has "
                          f"${client['Cash_Savings']:,} in cash savings.",
                "chart_base64": None,
            }

    # ----- 3) fallback → Groq LLM (OpenAI-compatible client) -----
    try:
        import openai

        openai.api_key = os.environ["OPENAI_API_KEY"]          # your gsk_ key
        openai.api_base = os.environ.get(
            "OPENAI_BASE_URL", "https://api.groq.com/openai/v1"
        )

        resp = openai.ChatCompletion.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": query.question}],
            max_tokens=512,
        )
        answer_text = resp.choices[0].message.content.strip()

        return {"answer": answer_text, "chart_base64": None}

    except Exception as e:
        # as a last resort try the local Ollama model if it’s bundled
        try:
            ollama_resp = subprocess.run(
                ["ollama", "run", "llama3", query.question],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {"answer": ollama_resp.stdout.strip(), "chart_base64": None}
        except Exception:
            return {"answer": f"AI fallback failed: {e}", "chart_base64": None}
