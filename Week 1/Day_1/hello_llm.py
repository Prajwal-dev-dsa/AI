import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"

prompt = "Hello, how are you?"

role = "user"

message = [
    {
        "role": role,
        "content": prompt
    }
]


response = client.chat.completions.create(
    model=model,
    messages=message
)

print(response.choices[0].message.content)