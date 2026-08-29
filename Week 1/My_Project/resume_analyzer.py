import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=my_api_key)

model="openai/gpt-oss-20b"

def extract_resume_text(file_path):
    """
    Opens a PDF file in binary mode and extracts all available text.
    """
    extracted_text = ""
    
    # Open the PDF file using file handling in read-binary ('rb') mode
    with open(file_path, 'rb') as pdf_file:
        
        # Initialize the PDF reader object to parse the binary data
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Iterate through all the pages in the document
        for page in pdf_reader.pages:
            
            # Extract text from the current page
            text = page.extract_text()
            
            # If text is successfully found on the page, append it
            if text:
                extracted_text += text + "\n"
                
    return extracted_text


system_prompt="""
You are an expert Technical Recruiter and Resume Analyzer AI. 
Your objective is to evaluate a candidate's resume against a specific Job Description (JD) and provide a strict, objective match analysis.

Instructions:
1. Compare the provided Resume Text against the Job Description.
2. Calculate a "matchPercentage" (0-100) based on the alignment of technical skills, production experience, and core requirements.
3. Identify key matching qualifications present in both texts.
4. Identify critical missing qualifications or skill gaps.
5. You MUST return your response exclusively as a valid, raw JSON object. Do not include markdown code blocks (like ```json), conversational text, or preambles. Use this exact schema:

{
  "matchPercentage": 85,
  "matchingKeywords": ["React", "Node.js", "REST APIs"],
  "missingKeywords": ["GraphQL", "AWS Lambda"],
  "justification": "The candidate has strong backend experience but lacks the specific cloud infrastructure requirements."
}
"""

system_message={
    "role": "system",
    "content": system_prompt
}

job_description="""
Job Title: Full-Stack Next.js Developer
Location: Remote
Experience: 0-2 Years / Entry Level to Mid

Job Overview:
We are seeking a Full-Stack Engineer skilled in modern TypeScript frameworks to build scalable web applications. You will design full-stack features, integrate payment flows, and implement real-time communication modules.

Key Requirements:
- Proficiency with Next.js, React.js, TypeScript, and Node.js.
- Hands-on experience with PostgreSQL, MongoDB, and ORMs like Prisma.
- Experience implementing WebSocket / Socket.IO connections for real-time features.
- Familiarity with third-party APIs (Stripe, Razorpay, OAuth providers).
- Strong foundation in Data Structures & Algorithms.
"""

resume_text = extract_resume_text('Resume.pdf')

user_prompt=f"""
Please evaluate the following candidate resume against the provided Job Description.

JOB DESCRIPTION: {job_description}

RESUME TEXT:
{resume_text}
"""

user_message={
    "role": "user",
    "content": user_prompt
}

messages=[system_message, user_message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
)

print(response.choices[0].message.content)