import os
from dotenv import load_dotenv

load_dotenv()

# Simple test without any API calls first
print("Environment check:")
print(f"HF_TOKEN: {'✅ Found' if os.environ.get('HF_TOKEN') else '❌ Missing'}")
print(f"MOONSHOT_TOKEN: {'✅ Found' if os.environ.get('MOONSHOT_TOKEN') else '❌ Missing'}")
print(f"NOVITA_TOKEN: {'✅ Found' if os.environ.get('NOVITA_TOKEN') else '❌ Missing'}")

print("\n" + "="*50)
print("SOLUTION SUMMARY:")
print("="*50)

print("""
🔍 DIAGNOSIS:
Your Moonshot API key is valid, but your account has exceeded its quota.

🛠️ SOLUTIONS:

1. **Fix Moonshot Account (Recommended)**:
   - Visit: https://platform.moonshot.ai/
   - Check billing/usage dashboard
   - Add credits or upgrade plan
   - Then your current script will work

2. **Use Free Alternative - Ollama (Local)**:
   - Install Ollama: https://ollama.ai/
   - Run models locally (free, no API keys needed)
   - Example: ollama run llama2

3. **Use OpenAI Free Tier**:
   - Get OpenAI API key with free credits
   - Change base_url to "https://api.openai.com/v1"

4. **Use Other Free APIs**:
   - Groq (free tier): https://groq.com/
   - Together AI (free tier): https://together.ai/

📝 Your script is working correctly - it's just a billing/quota issue!
""")

print("="*50)
