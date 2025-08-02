import os
import warnings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

load_dotenv()

# Set USER_AGENT environment variable to avoid warnings
os.environ.setdefault('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.environ.get("HF_TOKEN")
)

# Create chat instance for better text generation
chat = ChatHuggingFace(llm=llm, verbose=True)

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to ChromaDB
try:
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_client.heartbeat()
    print("✅ Connected to ChromaDB server")
except Exception as e:
    print(f"❌ Failed to connect to ChromaDB server: {e}")
    print("Please start ChromaDB server with : chroma run --host localhost --port 8000")
    exit(1)

# Initialize Chroma vector store
try:
    vector_store = Chroma(
        client=chroma_client,
        collection_name="embd_store_cover_letter",
        embedding_function=embedding_function
    )
    print("✅ Initialized Chroma vector store")
except Exception as e:
    print(f"❌ Failed to initialize Chroma vector store: {e}")
    exit(1)

store_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Store this information: {text}"
)

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

""" Splits text into chunks of max 100 words """
def split_text_into_chunks(text, chunk_size=5000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10)
    chunks = splitter.split_text(text)
    return chunks

""" Inserts text chunks from a PDF into ChromaDB """
def insert_text_from_pdf():
    pdf_path = input("Enter path to PDF file including the file name ending with .pdf: ")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Split text into smaller chunks
    chunks = split_text_into_chunks(extracted_text)
    
    # Generate embeddings for each chunk and store them
    for chunk in chunks:
        print(f"\nchunk : {chunk}")
        formatted_chunk = store_prompt_template.format(text=chunk)
        vector_store.add_texts([formatted_chunk])

    print(f"Text successfully split into {len(chunks)} chunks and added to embd_store_cover_letter from PDF.")

def scrape_job_with_langchain(url):
    """Alternative method using LangChain's WebBaseLoader"""
    try:
        print(f"Using LangChain WebLoader for: {url}")
        
        # Configure WebBaseLoader with custom headers
        loader = WebBaseLoader(
            web_paths=[url],
            header_template={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        docs = loader.load()
        
        if docs:
            content = docs[0].page_content
            # Clean and extract relevant content
            job_description = extract_job_relevant_content(content)
            print(f"✅ Successfully extracted with LangChain ({len(job_description)} characters)")
            return job_description
        else:
            print("❌ No content found with LangChain WebLoader")
            return None
            
    except Exception as e:
        print(f"❌ LangChain WebLoader failed: {e}")
        return None

def scrape_job_description_from_url(url):
    """Scrape job description from a given URL with fallback methods"""
    # Try LangChain WebBaseLoader first
    job_description = scrape_job_with_langchain(url)
    
    # If LangChain fails, try BeautifulSoup
    if not job_description:
        print("Trying alternative scraping method...")
        job_description = scrape_job_with_beautifulsoup(url)
    
    return job_description

def scrape_job_with_beautifulsoup(url):
    """Scrape job description using BeautifulSoup"""
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Fetching content with BeautifulSoup from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text (remove extra whitespace, newlines)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Try to extract job-relevant sections
        job_description = extract_job_relevant_content(text)
        
        if len(job_description) > 100:  # Ensure we got meaningful content
            print(f"✅ Successfully extracted job description ({len(job_description)} characters)")
            return job_description
        else:
            print("⚠️ Extracted content seems too short. Using full page content.")
            return text[:3000]  # Limit to first 3000 characters
            
    except requests.RequestException as e:
        print(f"❌ Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"❌ Error parsing content: {e}")
        return None

def extract_job_relevant_content(text):
    """Extract job-relevant sections from scraped text"""
    # Common job posting keywords to look for
    job_keywords = [
        'job description', 'responsibilities', 'requirements', 'qualifications',
        'duties', 'skills', 'experience', 'education', 'about the role',
        'what you will do', 'what we are looking for', 'ideal candidate',
        'key responsibilities', 'required skills', 'preferred qualifications'
    ]
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Find relevant sections
    relevant_sections = []
    
    for keyword in job_keywords:
        # Find keyword and extract surrounding context
        keyword_pos = text_lower.find(keyword)
        if keyword_pos != -1:
            # Extract a chunk around the keyword (500 chars before and after)
            start = max(0, keyword_pos - 500)
            end = min(len(text), keyword_pos + 1500)
            section = text[start:end].strip()
            if section and len(section) > 50:
                relevant_sections.append(section)
    
    if relevant_sections:
        # Combine all relevant sections
        combined = ' '.join(relevant_sections)
        # Remove duplicates and limit length
        return combined[:4000]  # Limit to 4000 characters
    else:
        # If no specific sections found, return first part of text
        return text[:2000]

def get_job_description():
    """Get job description from user input (URL or manual entry)"""
    print("\nHow would you like to provide the job description?")
    print("1. Enter a job posting URL (we'll extract it automatically)")
    print("2. Enter job description manually")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        url = input("\nEnter the job posting URL: ").strip()
        if url:
            job_description = scrape_job_description_from_url(url)
            if job_description:
                print("\n" + "="*50)
                print("EXTRACTED JOB DESCRIPTION PREVIEW:")
                print("="*50)
                print(job_description[:500] + "..." if len(job_description) > 500 else job_description)
                print("="*50)
                
                confirm = input("\nDoes this look correct? (y/n): ").strip().lower()
                if confirm == 'y':
                    return job_description
                else:
                    print("Let's try manual entry instead...")
                    return get_manual_job_description()
            else:
                print("Failed to extract job description from URL. Let's try manual entry...")
                return get_manual_job_description()
        else:
            print("No URL provided. Let's try manual entry...")
            return get_manual_job_description()
    
    elif choice == "2":
        return get_manual_job_description()
    
    else:
        print("Invalid choice. Using manual entry...")
        return get_manual_job_description()

def get_manual_job_description():
    """Get job description through manual input"""
    print("\nPlease enter the job description:")
    print("(Tip: You can paste multiple lines. Press Enter twice when done)")
    
    lines = []
    while True:
        line = input()
        if line == "" and lines:  # Empty line and we have content
            break
        lines.append(line)
    
    job_description = "\n".join(lines)
    return job_description if job_description.strip() else "No job description provided."

def fetch_relevant_resume_content(job_description):
    """Fetch relevant resume content based on job description"""
    # Create search query from job description keywords
    search_query = f"Find relevant experience and skills related to: {job_description}"
    
    # Search for relevant chunks in vector store
    results = vector_store.similarity_search(search_query, k=6)  # Get top 6 relevant chunks
    
    if results:
        relevant_content = "\n".join([doc.page_content for doc in results])
        print(f"\nRelevant resume content found: {len(results)} chunks")
        return relevant_content
    else:
        print("No relevant content found in resume.")
        return None

def generate_cover_letter(resume_content, job_description):
    """Generate tailored cover letter using AI"""
    prompt = f"""
    You are a professional cover letter writer. Based on the candidate's resume content and the job description provided, 
    write a compelling cover letter that:
    
    1. Highlights relevant experience from the resume that matches the job requirements
    2. Shows enthusiasm for the specific role and company
    3. Demonstrates how the candidate's skills align with the job description
    4. Maintains a professional yet engaging tone
    5. Is concise (3-4 paragraphs)
    
    Resume Content:
    {resume_content}
    
    Job Description:
    {job_description}
    
    Please write a tailored cover letter:
    """
    
    try:
        response = chat.invoke(prompt)
        # Extract the actual content from the response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        print(f"Error generating cover letter: {e}")
        return None

def main_cover_letter_agent():
    """Main function to run the cover letter generation process"""
    print("=== Cover Letter AI Agent ===")
    
    # Step 1: Get and store resume
    print("\n1. First, let's process your resume...")
    insert_text_from_pdf()
    
    # Step 2: Get job description
    print("\n2. Now, please provide the job description...")
    job_description = get_job_description()
    
    # Step 3: Find relevant resume content
    print("\n3. Analyzing resume against job requirements...")
    relevant_resume = fetch_relevant_resume_content(job_description)
    
    if relevant_resume:
        # Step 4: Generate tailored cover letter
        print("\n4. Generating your tailored cover letter...")
        cover_letter = generate_cover_letter(relevant_resume, job_description)
        
        if cover_letter:
            print("\n" + "="*50)
            print("YOUR TAILORED COVER LETTER:")
            print("="*50)
            print(cover_letter)
            print("="*50)
            
            # Option to save to file
            save_option = input("\nWould you like to save this cover letter to a file? (y/n): ")
            if save_option.lower() == 'y':
                filename = input("Enter filename (without extension): ") + ".txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(cover_letter)
                print(f"Cover letter saved to {filename}")
        else:
            print("Failed to generate cover letter. Please try again.")
    else:
        print("Could not find relevant resume content. Please check your resume PDF.")

# User selection menu
while True:
    print("\n" + "="*50)
    print("Welcome to Cover Letter Tailoring Agent!")
    print("="*50)
    choice = input("\nChoose an action:\n1. Generate Cover Letter\n2. Exit\nEnter choice: ")
    
    if choice == "1":
        main_cover_letter_agent()
    elif choice == "2":
        print("All the best!")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")