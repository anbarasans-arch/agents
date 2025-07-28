import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

moonshot_token = os.environ.get("MOONSHOT_TOKEN")
if not moonshot_token:
    raise ValueError("MOONSHOT_TOKEN not found in environment variables. Please check your .env file.")

client = OpenAI(
    #base_url="https://router.huggingface.co/v1",
    base_url="https://api.moonshot.ai/v1",
    api_key=moonshot_token,
    #api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ],
)

print(completion.choices[0].message)