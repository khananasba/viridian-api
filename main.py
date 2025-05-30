from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import subprocess

app = FastAPI()

# ✅ Enable CORS to allow frontend (like Lovable) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your actual frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your client CSV
df = pd.read_csv("mock_wealth_clients_with_names.csv")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_ai(query: Query):
    question = query.question.lower()
    name_match = [name for name in df['Name'] if name.lower() in question]
    
    if not name_match:
        return {"answer": "Sorry, I couldn't find any matching client."}
    
    client = df[df['Name'] == name_match[0]].iloc[0]
    
    # Simple examples
    if "income" in question:
        return {"answer": f"{client['Name']}'s annual income is ${client['Annual_Income']:,}."}
    
    if "super" in question:
        return {"answer": f"{client['Name']}'s super balance is ${client['Super_Balance']:,}."}

    if "cash" in question or "savings" in question:
        return {"answer": f"{client['Name']} has ${client['Cash_Savings']:,} in cash savings."}

    # fallback: send the question to Ollama LLaMA3
    try:
        ollama_response = subprocess.run(
            ["ollama", "run", "llama3", f"{query.question}"],
            capture_output=True,
            text=True
        )
        return {"answer": ollama_response.stdout.strip()}
    except Exception as e:
        return {"answer": "AI fallback failed. Error: " + str(e)}
