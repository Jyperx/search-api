import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("../admin/.env")
genai.configure(api_key=os.getenv("VITE_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)
