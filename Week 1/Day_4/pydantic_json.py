import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

client=Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"

user_text="hello, my name is prajwal. I am a student of BCA. i purchased iphone 15 pro max yesterday from your store and its not working now, please contact me on my email prajwal77dwivedi@gmail.com or phone number 9876543210. i am from Delhi."

user_prompt = f"""
this is a customer ticket please extract personal infromation from this ticket {user_text}
"""

user_message = {
    "role": "user",
    "content": user_prompt
}

from pydantic import BaseModel

class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema=Ticket.model_json_schema()

response_format={"type":"json_object"}

system_prompt=f"""
You are a customer support agent. Extract strictly only the following information from the user's message which is provided in this schema: {schema}

Respond with a JSON object containing these fields.
"""

system_message={
    "role":"system",
    "content":system_prompt
}

messages=[system_message, user_message]

response=client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format,
)

answer=response.choices[0].message.content
print(answer)


import json
raw_text=answer
data_file=json.loads(raw_text)
ticket=Ticket(**data_file)
print(data_file)
print(ticket)