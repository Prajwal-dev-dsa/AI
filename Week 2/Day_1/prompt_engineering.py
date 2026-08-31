import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

model="openai/gpt-oss-20b"

def llm_response(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages=[message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=2
    )
    return response.choices[0].message.content


prompt="""

ROLE: You are an expert customer support ticket classifier.

TASK: Your task is to classify the customer's message into the most appropriate support category.

CONSTRAINTS:
- Choose only one category.
- Do not invent or create new categories.
- Classify the message based only on the information provided.
- Do not try to solve the customer's problem.
- Do not ask the customer for additional information.
- Keep the response short and classification-focused.

OUTPUT FORMAT:
Return only one category from this list:

- billing
- technical
- account
- shipping
- return

Return the category as plain text.
Do not include explanations, labels, or extra text.

EXAMPLES:

Customer:
"I was charged twice for the same order."

Output:
billing

Customer:
"My package has not arrived yet."

Output:
shipping

Customer:
"I forgot my password and cannot log in."

Output:
account

Customer:
"The checkout page keeps showing an error."

Output:
technical

FALLBACK:
- If the customer's message does not clearly fit any of the
  available categories, return "other".
- Never create a new category.
- If multiple categories seem possible, choose the most relevant one.

MESSAGE: i want to refund my phone
"""

print(llm_response(prompt))