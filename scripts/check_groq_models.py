"""Quick check: what models does your Groq API key have access to?"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

print("Models available on your Groq key:\n")
for m in client.models.list().data:
    print(f"  {m.id}")
