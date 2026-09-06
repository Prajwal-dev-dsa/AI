import os
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

model="openai/gpt-oss-20b"

def llm_call(system_prompt, user_prompt):
    system_message = {
        "role":"system",
        "content":system_prompt
    }
    user_message = {
        "role":"user",
        "content":user_prompt
    }
    messages=[system_message, user_message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def step1_extract_skills(resume_text):
    print("Extracting skills from resume...")
    system_prompt = "You are a professional HR assistant. Extract the skills from the resume. Return only the skills comma separated. Do not include any other text. Do not include any formatting. Do not guess skills if they are not in the resume. If you cannot find any skills, return an empty string."
    user_prompt = resume_text
    return llm_call(system_prompt, user_prompt)


def step2_extract_jd_skills(job_description):
    print("Extracting skills from job description...")
    system_prompt = "You are a professional HR assistant. Extract the skills from the job description. Return only the skills comma separated. Do not include any other text. Do not include any formatting. Do not guess skills if they are not in the job description. If you cannot find any skills, return an empty string."
    user_prompt = job_description
    return llm_call(system_prompt, user_prompt)


def step3_match_skills(resume_skills, jd_skills):
    print("Matching skills...")
    system_prompt = "You are a professional HR assistant. Match the skills from the resume with the skills from the job description. After matching, calculate the match percentage and return the result in the following format: 'X%'. Also give a reason for the match percentage. And at last also tell whether should I call this candidate for an interview or reject."
    user_prompt = f"Resume Skills: {resume_skills}\nJob Description Skills: {jd_skills}"
    return llm_call(system_prompt, user_prompt)


jd = """
    Job Title: Software Developer

    Company: Tech Solutions

    Job Description:
    We are looking for a Software Developer to join our team.

    Responsibilities:
    - Develop and maintain software applications.
    - Write clean and efficient code.
    - Debug and fix software issues.
    - Work with the development team.

    Requirements:
    - Good knowledge of Python.
    - Basic understanding of databases.
    - Knowledge of Git.
    - Good problem-solving skills.
    - Ability to work in a team.

    Experience: 0-2 years

    Location: Remote
"""

resume = """
    JOHN DOE
    Software Developer

    Email: john.doe@email.com
    Phone: +91 9876543210
    Location: Bangalore, India
    LinkedIn: linkedin.com/in/johndoe
    GitHub: github.com/johndoe

    SUMMARY
    Computer Science graduate with a strong foundation in software development.
    Experienced in building web applications using Python, JavaScript, and React.
    Good problem-solving and communication skills.

    EDUCATION
    Bachelor of Technology in Computer Science
    ABC Institute of Technology
    2022 - 2026
    CGPA: 8.5/10

    SKILLS
    - Python
    - JavaScript
    - React.js
    - Node.js
    - Express.js
    - MongoDB
    - SQL
    - Git
    - REST APIs

    EXPERIENCE
    Software Developer Intern
    Tech Solutions Pvt. Ltd.
    Jan 2026 - Jun 2026

    - Developed web applications using Python and JavaScript.
    - Built and tested REST APIs.
    - Worked with MongoDB and SQL databases.
    - Fixed bugs and improved application performance.
    - Collaborated with other developers using Git.

    PROJECTS
    Task Management Application
    - Built a full-stack task management application.
    - Implemented user authentication and CRUD operations.
    - Used React.js, Node.js, Express.js, and MongoDB.

    Personal Portfolio
    - Developed a responsive portfolio website.
    - Added projects, skills, education, and contact sections.
    - Used React.js and CSS.

    CERTIFICATIONS
    - Python Programming Certificate
    - Full Stack Web Development Certificate

    ACHIEVEMENTS
    - Solved 300+ coding problems on LeetCode.
    - Participated in multiple coding competitions.
    - Built several personal software projects.

    LANGUAGES
    - English
    - Hindi
"""

def main(resume_text, job_description):
    resume_skills = step1_extract_skills(resume_text)
    sleep(3)
    jd_skills = step2_extract_jd_skills(job_description)
    sleep(3)
    final_verdict = step3_match_skills(resume_skills, jd_skills)
    return final_verdict


print("Final Verdict:", main(resume, jd))