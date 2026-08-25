import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"

systemPrompt={
    "role":"system",
    "content":"You are my girlfriend. So behave like my beautiful girlfriend."
}

humanPrompt={
    "role":"user",
    "content":"Hello baby! I love you so much! How are you?"
}

messages=[systemPrompt, humanPrompt]

response=client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=2
)

print(response.choices[0].message.content)