import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    for m in genai.list_models():
        if "generateContent" in (m.supported_generation_methods or []):
            print(f"Dostupný model: {m.name}")
except Exception as e:
    print(f"Chyba: {e}", file=sys.stderr)
    sys.exit(1)
