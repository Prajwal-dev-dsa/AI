import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

client=Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"

prompt1="hiii"
prompt2="what is ai?"
prompt3="explain javascript in 500 words"

prompt=[prompt1, prompt2, prompt3]


for p in prompt:
    message={
        "role":"user",
        "content":p
    }
    messages=[message]
    response=client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100
    )
    usage=response.usage
    print("Prompt response:")
    print(response.choices[0].message.content)
    print(f"Tokens used:")
    print(f"User token: {usage.prompt_tokens}")
    print(f"Response token: {usage.completion_tokens}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Total token: {usage.total_tokens}\n\n")