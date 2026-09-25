import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model="qwen/qwen3.8-27b"

client=Groq(api_key=my_api_key)

context_dict={
    "name": "Name is Prajwal Dwivedi",
    "email": "Email is prajwal77dwivedi@gmail.com",
    "phone": "Phone is 9876543210",
    "address": "Address is Delhi, India",
    "age": "Age is 25",
}

def match_context(user_prompt):
    for key, value in context_dict.items():
        if key in user_prompt.lower():
            return value
    return "no context matched of the user prompt, kindly say that your question doesnt have any context for me"

def ask_llm(user_prompt):
    matched_context = match_context(user_prompt)
    sys_prompt=f"""answer in one line. if question matches with the provided context then use the context to answer the user's question. context: {matched_context}"""

    sys_message={
        "role": "system",
        "content": sys_prompt
    }
    user_message={
        "role": "user",
        "content": user_prompt
    }
    response=client.chat.completions.create(
        model=model,
        messages=[sys_message, user_message]
    )
    return response.choices[0].message.content

print(ask_llm("how old are you?"))