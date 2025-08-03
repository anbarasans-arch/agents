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
import streamlit as st
import tempfile
import time
# Document processing libraries
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    st.warning("⚠️ python-docx not installed. DOCX support will be limited.")
    DOCX_AVAILABLE = False

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    st.warning("⚠️ docx2txt not installed. Alternative DOCX processing will be limited.")
    DOCX2TXT_AVAILABLE = False

# Suppress specific warnings
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

load_dotenv()

# Set USER_AGENT environment variable to avoid warnings
os.environ.setdefault('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

# Streamlit page configuration
st.set_page_config(
    page_title="AI Cover Letter Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize LLM and other components only once using Streamlit session state
@st.cache_resource
def initialize_llm():
    """Initialize LLM with caching"""
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )
    return ChatHuggingFace(llm=llm, verbose=False)

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings with caching"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def initialize_chromadb():
    """Initialize ChromaDB connection with caching"""
    try:
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        chroma_client.heartbeat()
        
        embedding_function = initialize_embeddings()
        vector_store = Chroma(
            client=chroma_client,
            collection_name="embd_store_cover_letter",
            embedding_function=embedding_function
        )
        return chroma_client, vector_store, True
    except Exception as e:
        return None, None, False

# Initialize components
store_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Store this information: {text}"
)

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        extracted_text = "\n".join(doc.page_content for doc in docs)
        return extracted_text
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        extracted_text = ""
        
        # Method 1: Using python-docx
        if DOCX_AVAILABLE:
            try:
                doc = Document(file_path)
                text_content = []
                for paragraph in doc.paragraphs:
                    text_content.append(paragraph.text)
                extracted_text = "\n".join(text_content)
            except Exception as e:
                st.warning(f"⚠️ python-docx method failed: {e}")
        
        # Method 2: If the first method doesn't get enough content, try docx2txt
        if (len(extracted_text.strip()) < 100) and DOCX2TXT_AVAILABLE:
            try:
                extracted_text = docx2txt.process(file_path)
            except Exception as e:
                st.warning(f"⚠️ docx2txt method failed: {e}")
        
        # If neither method worked well
        if len(extracted_text.strip()) < 50:
            st.error("❌ Could not extract sufficient text from DOCX file.")
            st.info("💡 Try converting your file to PDF format for better compatibility.")
            return None
        
        return extracted_text
    except Exception as e:
        st.error(f"❌ Error extracting text from DOCX: {str(e)}")
        return None

def extract_text_from_doc(file_path):
    """Extract text from DOC file"""
    try:
        # For .doc files, we'll use a simple approach
        # Note: This might not work for all .doc files due to format complexity
        st.warning("⚠️ .DOC file support is limited. For best results, please convert to .DOCX or .PDF format.")
        
        # Try to read as text (this is a fallback and may not work well)
        with open(file_path, 'rb') as file:
            content = file.read()
            # Try to decode and extract readable text
            try:
                text = content.decode('utf-8', errors='ignore')
                # Clean up the text to remove binary artifacts
                import string
                printable = set(string.printable)
                cleaned_text = ''.join(filter(lambda x: x in printable, text))
                # Remove excessive whitespace
                cleaned_text = ' '.join(cleaned_text.split())
                return cleaned_text if len(cleaned_text) > 50 else None
            except:
                return None
    except Exception as e:
        st.error(f"❌ Error extracting text from DOC: {str(e)}")
        st.error("💡 Tip: Try converting your .doc file to .docx or .pdf format for better compatibility.")
        return None

def process_document_file(uploaded_file):
    """Process uploaded document file (PDF, DOCX, or DOC) and store in ChromaDB"""
    chroma_client, vector_store, connected = initialize_chromadb()
    
    if not connected:
        st.error("❌ Failed to connect to ChromaDB. Please start the server first.")
        return False
    
    try:
        # Get file extension
        file_name = uploaded_file.name
        file_extension = file_name.lower().split('.')[-1]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract text based on file type
        extracted_text = None
        
        if file_extension == 'pdf':
            with st.spinner("📄 Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(tmp_file_path)
        elif file_extension == 'docx':
            with st.spinner("📄 Extracting text from DOCX..."):
                extracted_text = extract_text_from_docx(tmp_file_path)
        elif file_extension == 'doc':
            with st.spinner("📄 Extracting text from DOC..."):
                extracted_text = extract_text_from_doc(tmp_file_path)
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return False
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            st.error("❌ Could not extract sufficient text from the document. Please check the file or try a different format.")
            return False
        
        # Show preview of extracted text
        with st.expander("👀 Preview Extracted Text"):
            st.text_area("Extracted Content Preview", extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text, height=150, disabled=True)
        
        # Split into chunks
        with st.spinner("✂️ Splitting text into chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=10)
            chunks = splitter.split_text(extracted_text)
        
        # Store in vector database
        with st.spinner("💾 Storing in vector database..."):
            for chunk in chunks:
                formatted_chunk = store_prompt_template.format(text=chunk)
                vector_store.add_texts([formatted_chunk])
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        st.success(f"✅ Successfully processed {file_extension.upper()} file! Split into {len(chunks)} chunks and stored in database.")
        st.info(f"📊 Extracted {len(extracted_text)} characters of text content.")
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return False

def scrape_job_description_ui(url):
    """Scrape job description with UI feedback"""
    try:
        with st.spinner("🔍 Extracting job description from URL..."):
            # Try LangChain WebBaseLoader first
            job_description = scrape_job_with_langchain(url)
            
            # If LangChain fails, try BeautifulSoup
            if not job_description:
                st.info("Trying alternative scraping method...")
                job_description = scrape_job_with_beautifulsoup(url)
            
            return job_description
    except Exception as e:
        st.error(f"❌ Error scraping URL: {str(e)}")
        return None

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
    chroma_client, vector_store, connected = initialize_chromadb()
    
    if not connected:
        return None
    
    # Create search query from job description keywords
    search_query = f"Find relevant experience and skills related to: {job_description}"
    
    # Search for relevant chunks in vector store
    results = vector_store.similarity_search(search_query, k=6)  # Get top 6 relevant chunks
    
    if results:
        relevant_content = "\n".join([doc.page_content for doc in results])
        return relevant_content
    else:
        return None

def generate_cover_letter(resume_content, job_description):
    """Generate tailored cover letter using AI"""
    chat = initialize_llm()
    
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
            cover_letter = response.content
        else:
            cover_letter = str(response)
        
        # Clean up the response - remove any unwanted prefixes or formatting
        cover_letter = cover_letter.strip()
        
        # Remove common AI response prefixes if they exist
        prefixes_to_remove = [
            "Here's a tailored cover letter:",
            "Here is a tailored cover letter:",
            "Dear Hiring Manager,",
            "Based on the provided information,"
        ]
        
        for prefix in prefixes_to_remove:
            if cover_letter.lower().startswith(prefix.lower()):
                cover_letter = cover_letter[len(prefix):].strip()
        
        return cover_letter
        
    except Exception as e:
        st.error(f"Error generating cover letter: {e}")
        return None

def main_streamlit_app():
    """Main Streamlit application"""
    # App header
    st.title("🎯 AI-Powered Cover Letter Generator")
    st.markdown("*Generate tailored cover letters that match your resume to specific job descriptions*")
    
    # Feature highlight
    st.info("🆕 **New Feature**: Now supports PDF, DOCX, and DOC resume formats!")
    
    # Sidebar for ChromaDB status
    with st.sidebar:
        st.header("🔧 System Status")
        chroma_client, vector_store, connected = initialize_chromadb()
        
        if connected:
            st.success("✅ ChromaDB Connected")
        else:
            st.error("❌ ChromaDB Disconnected")
            st.warning("Please start ChromaDB server:")
            st.code("docker run -p 8000:8000 chromadb/chroma")
            st.stop()
        
        st.markdown("---")
        st.header("📄 Supported File Formats")
        st.markdown("""
        **Resume Upload:**
        - 📄 **PDF** - Fully supported
        - 📝 **DOCX** - Fully supported  
        - 📄 **DOC** - Limited support*
        
        *For best results with .DOC files, please convert to .DOCX or .PDF format.
        """)
        
        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        - Ensure your resume has clear text (not scanned images)
        - Use standard fonts for better text extraction
        - Break your resume into clear sections
        - Include keywords relevant to your target jobs
        """)
    
    # Initialize session state
    if "resume_processed" not in st.session_state:
        st.session_state.resume_processed = False
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""
    if "cover_letter" not in st.session_state:
        st.session_state.cover_letter = ""
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Upload Resume", "💼 Job Description", "📝 Generate Cover Letter"])
    
    # Tab 1: Resume Upload
    with tab1:
        st.header("Step 1: Upload Your Resume")
        st.markdown("Upload your resume in PDF, DOCX, or DOC format to extract and store your information.")
        
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=["pdf", "docx", "doc"],
            help="Upload a document file containing your resume (PDF, DOCX, or DOC)"
        )
        
        if uploaded_file is not None:
            # Show file info
            st.info(f"📎 Selected: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            if st.button("📄 Process Resume", type="primary"):
                if process_document_file(uploaded_file):
                    st.session_state.resume_processed = True
                    st.balloons()
        
        if st.session_state.resume_processed:
            st.success("✅ Resume has been processed and stored!")
    
    # Tab 2: Job Description
    with tab2:
        st.header("Step 2: Provide Job Description")
        
        input_method = st.radio(
            "How would you like to provide the job description?",
            ["📝 Manual Entry", "🔗 URL Extraction"],
            help="Choose to either paste the job description manually or extract it from a URL"
        )
        
        if input_method == "🔗 URL Extraction":
            url = st.text_input(
                "Job Posting URL",
                placeholder="https://example.com/job-posting",
                help="Enter the URL of the job posting"
            )
            
            if url and st.button("🔍 Extract Job Description"):
                job_description = scrape_job_description_ui(url)
                if job_description:
                    st.session_state.job_description = job_description
                    st.success("✅ Job description extracted successfully!")
                    
                    # Show preview
                    with st.expander("📋 Preview Extracted Content"):
                        st.text_area("Extracted Job Description", job_description, height=200, disabled=True)
                else:
                    st.error("❌ Failed to extract job description. Please try manual entry.")
        
        else:  # Manual Entry
            job_description = st.text_area(
                "Job Description",
                value=st.session_state.job_description,
                height=300,
                placeholder="Paste the job description here...",
                help="Copy and paste the complete job description including requirements and responsibilities"
            )
            
            if job_description:
                st.session_state.job_description = job_description
                st.success(f"✅ Job description saved! ({len(job_description)} characters)")
    
    # Tab 3: Generate Cover Letter
    with tab3:
        st.header("Step 3: Generate Your Cover Letter")
        
        # Check prerequisites
        if not st.session_state.resume_processed:
            st.warning("⚠️ Please upload and process your resume first (Step 1)")
            
        elif not st.session_state.job_description:
            st.warning("⚠️ Please provide a job description first (Step 2)")
            
        else:
            st.success("🎯 Ready to generate your cover letter!")
            
            if st.button("🚀 Generate Cover Letter", type="primary"):
                with st.spinner("🧠 AI is analyzing your resume against the job requirements..."):
                    # Fetch relevant resume content
                    relevant_resume = fetch_relevant_resume_content(st.session_state.job_description)
                    
                    if relevant_resume:
                        st.info(f"📊 Found relevant resume content for analysis")
                        
                        # Generate cover letter
                        with st.spinner("✍️ Writing your personalized cover letter..."):
                            cover_letter = generate_cover_letter(relevant_resume, st.session_state.job_description)
                            
                            if cover_letter:
                                st.session_state.cover_letter = cover_letter
                                st.success("🎉 Cover letter generated successfully!")
                                
                                # Debug: Show cover letter length and preview
                                st.info(f"📊 Generated cover letter with {len(cover_letter)} characters")
                                
                            else:
                                st.error("❌ Failed to generate cover letter. Please try again.")
                    else:
                        st.error("❌ Could not find relevant resume content. Please check if your resume was processed correctly.")
            
            # Display generated cover letter
            if st.session_state.cover_letter:
                st.markdown("---")
                st.subheader("📝 Your Tailored Cover Letter")
                
                # Show cover letter content length for debugging
                st.caption(f"Cover letter length: {len(st.session_state.cover_letter)} characters")
                
                # Primary display: Text area for easy reading and copying
                st.text_area(
                    "Generated Cover Letter",
                    value=st.session_state.cover_letter,
                    height=400,
                    help="Your personalized cover letter - you can copy the text from here"
                )
                
                # Alternative display: Code block for guaranteed visibility
                with st.expander("📋 Alternative View (Click to expand)"):
                    st.code(st.session_state.cover_letter, language=None)
                
                # Formatted display using markdown
                with st.expander("🎨 Formatted View (Click to expand)"):
                    # Split into paragraphs and display each one
                    paragraphs = st.session_state.cover_letter.split('\n\n')
                    for i, paragraph in enumerate(paragraphs):
                        if paragraph.strip():
                            st.markdown(f"**Paragraph {i+1}:**")
                            st.write(paragraph.strip())
                
                # Download button
                st.download_button(
                    label="💾 Download Cover Letter",
                    data=st.session_state.cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain",
                    help="Download your cover letter as a text file"
                )

# Run the Streamlit app
if __name__ == "__main__":
    main_streamlit_app()