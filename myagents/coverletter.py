from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import PyPDF2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Get HuggingFace token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not found in environment variables.")
    print("Please make sure your .env file contains: HF_TOKEN=your_huggingface_token")
    exit(1)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hf_token
)

chat = ChatHuggingFace(
    llm=llm, verbose=True, temperature=0.1)

user_question = input("What is the job description:")

# Option to upload PDF resume
resume_path = input("Enter path to your resume PDF (or press Enter to use default): ").strip()

# Use default path if no input provided
if not resume_path:
    resume_path = r"C:\Users\anbub\Documents\Interview\Anbarasan_StaffTPM.pdf"
    print(f"Using default resume: {resume_path}")

resume_text = ""

if resume_path and os.path.exists(resume_path):
    print("Extracting text from PDF...")
    resume_text = extract_text_from_pdf(resume_path)
    if resume_text:
        print("✅ Resume text extracted successfully!")
    else:
        print("❌ Failed to extract text from PDF")
        resume_text = ""
else:
    if resume_path:  # Path was provided but file doesn't exist
        print("❌ PDF file not found. Continuing without resume...")

# Prepare messages
system_prompt = """You are an expert in writing professional cover letters. 
You will create a personalized cover letter based on the job description provided.
If a resume is provided, use the relevant experience and skills from it to tailor the cover letter."""

if resume_text:
    human_prompt = f"""
Please write a professional cover letter for this job description:

JOB DESCRIPTION:
{user_question}

MY RESUME:
{resume_text}

Please create a tailored cover letter that highlights my relevant experience and skills from my resume that match the job requirements.
"""
else:
    human_prompt = f"Please write a professional cover letter for this job description: {user_question}"

messages = [
    ("system", system_prompt),
    ("human", human_prompt)
]

ai_msg = chat.invoke(messages)
print(ai_msg.content)