# ---------------------------- main.py ----------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # head-less backend for Render
import matplotlib.pyplot as plt
import io, base64, os, subprocess
import openai                   # works with Groq because of OPENAI_BASE_URL

# ╭───────────────────────────────╮
# │ FastAPI app + CORS middleware │
# ╰───────────────────────────────╯
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               # or ["https://your-lovable-site"]
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ╭──────────────────────╮
# │ Load client CSV once │
# ╰──────────────────────╯
df = pd.read_csv("mock_wealth_clients_with_names.csv")

# ╭─────────────╮
# │ Pydantic IO │
# ╰─────────────╯
class Query(BaseModel):
    question: str

# ╭────────────────────────────────────────────────────────╮
# │ Helper: make a bar chart and return it as base64 PNG   │
# ╰────────────────────────────────────────────────────────╯
def make_bar_chart(title: str, labels, values) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#4e79a7")
    ax.set_title(title)
    ax.set_ylabel("Amount ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ╭──────────────────────────────────────────╮
# │ Main endpoint used by Lovable `/ask`     │
# ╰──────────────────────────────────────────╯
@app.post("/ask")
def ask_ai(query: Query):
    q = query.question.lower()

    # 1️⃣  Find referenced client(s)
    matched_names = [name for name in df["Name"] if name.lower() in q]

    # ✦ Example analytic question: clients over a threshold
    if "cash" in q and "more than" in q and "client" in q:
        # crude parse: grab the first number that appears
        import re
        m = re.search(r"(\d[\d,\.]*)", q)
        if not m:
            raise HTTPException(400, "Could not detect threshold.")
        threshold = float(m.group(1).replace(",", ""))
        subset = df[df["Cash_Savings"] > threshold]

        if subset.empty:
            return {"answer": f"No clients have cash savings > ${threshold:,.0f}."}

        # Build answer text
        answer_lines = [
            f"{row.Name}: ${row.Cash_Savings:,.0f}" for row in subset.itertuples()
        ]
        # Optional chart
        chart_b64 = make_bar_chart(
            f"Clients with Cash > ${threshold:,.0f}",
            subset["Name"].tolist(),
            subset["Cash_Savings"].tolist(),
        )
        return {
            "answer": "• " + "\n• ".join(answer_lines),
            "chart_base64": chart_b64,
        }

    # 2️⃣  Simple per-client Q&A
    if matched_names:
        client = df[df["Name"] == matched_names[0]].iloc[0]

        if "income" in q:
            return {
                "answer": f"{client['Name']}'s annual income is ${client['Annual_Income']:,}.",
            }
        if "super" in q:
            return {
                "answer": f"{client['Name']}'s super balance is ${client['Super_Balance']:,}.",
            }
        if "cash" in q or "savings" in q:
            return {
                "answer": f"{client['Name']} has ${client['Cash_Savings']:,} in cash savings."
            }

        # Example chart request
        if "chart" in q and "finance" in q:
            labels = ["Income", "Super", "Cash"]
            values = [
                client["Annual_Income"],
                client["Super_Balance"],
                client["Cash_Savings"],
            ]
            chart_b64 = make_bar_chart(f"{client['Name']}'s Finances", labels, values)
            return {
                "answer": f"Here’s {client['Name']}'s financial breakdown.",
                "chart_base64": chart_b64,
            }

    # 3️⃣  Fallback to Groq / OpenAI-compatible LLM
    try:
        openai.api_key  = os.getenv("OPENAI_API_KEY")
        openai.base_url = os.getenv("OPENAI_BASE_URL")  # e.g. https://api.groq.com/openai/v1
        resp = openai.chat.completions.create(
            model="mixtral-8x7b-32768",                # or "llama3-70b-8192"
            messages=[{"role": "user", "content": query.question}],
            max_tokens=512,
        )
        return {"answer": resp.choices[0].message.content.strip()}
    except Exception as e:
        # last-ditch: try local Ollama (if running) – optional
        try:
            res = subprocess.run(
                ["ollama", "run", "llama3", query.question],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return {"answer": res.stdout.strip() or "No response from Ollama."}
        except Exception:
            raise HTTPException(500, f"LLM fallback failed: {e}")

# ---------------------------- end main.py ----------------------------
