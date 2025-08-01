"""
docker run -p 8000:8000 chromadb/chroma
py 14a-Fraud-detect-hgface-llm-chromadb-langchain-insert-fetch-chunks-pdf-interactive-pdf.py

pip install module_that_the_error_msg_shows

"""
#C:\Users\anbub\Documents\agents\AGENTS\support\CardTransactions_Genuine_AdamThorpe.pdf
#C:\Users\anbub\Documents\agents\AGENTS\support\CardTransactions_Genuine_JohnMayo.pdf


from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Hugging Face LLM
# setx huggingfacehub_api_token hf_CzXXXXX
llm = HuggingFaceEndpoint(
    #repo_id="Qwen/Qwen3-30B-A3B",
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.environ.get("HF TOKEN")
)


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
    collection_name="embd_store_4",
    embedding_function=embedding_function
    )
    print("✅ Initialized Chroma vector store")
except Exception as e:
    print(f"❌ Failed to initialize Chroma vector store: {e}")
    exit(1)

# Define structured prompt templates
store_prompt_template = PromptTemplate(
    input_variables=["text"],
    #template="Store this information: {text}"
	template="{text}"
)

search_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Find relevant information related to: {query}"
)

# Create a chat instance
chat = ChatHuggingFace(llm=llm, verbose=True)


""" Extracts text from a PDF using LangChain's PyPDFLoader """
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
    
    print(f"Text successfully split into {len(chunks)} chunks and added to embd_store_4 from PDF.")

""" Fetches text from ChromaDB that semantically matches user input """
def fetch_matching_text():
    query_text = input("Enter search query: ")
    formatted_query = search_prompt_template.format(query=query_text)
    print(f"\n {formatted_query}")

    # Retrieve top k matches
    results = vector_store.similarity_search(formatted_query, k=1)  
    
    if results:
        relevant_text = results[0].page_content
        #relevant_text = "Adam Thorpe made an in person genuine transaction of USD 10.00 on 1-Jan-2024 to buy Milk at the location Hong Kong, China. Adam Thorpe made an in person genuine transaction of USD 25.00 on 1-Jan-2024 to buy Poultry at the location Hong Kong, China.Adam Thorpe made an in person genuine transaction of USD 15.00 on 1-Jan-2024 to buy Zara leather jacket at the location Hong Kong, China.Adam Thorpe made an in person genuine transaction of USD 300.00 on 1-Jan-2024 to buy Water bottles at the location Hong Kong, China.Adam Thorpe made an in person genuine transaction of USD 402 on 1-Jan-2024 to buy Xbox at the location Hong Kong, China."

        print("\nMatching Result:", relevant_text)
        return relevant_text
    else:
        print("No matching results found.")
        return None

""" Interacts with ai LLM using retrieved text """
def chat_with_ai():
    relevant_text = fetch_matching_text()
    if relevant_text:
        system_msg = "You are an AI-powered fraud detection system."
        user_question = input("\nEnter your question about the retrieved text: ")

#detailed conclusive prompt with chain of thought
        
        user_question = user_question + " Your task is to determine if this transaction appears to be fraud. You will be provided this customers recent genuine payment transactions. You will analyze and understand the spending patterns. Then you will classify the new transaction as fraud or genuine strictly based on the following approach: 1- From the new transaction, extract the item name, cost and location. 2- From the spending patterns provided, create a list of a. item names, b. purchase category type, c. cost of the item and d. location of the purchase. 3- Next check if the new transaction item falls in one of the categories created above. If it does not then flag this transaction as fraud and provide your response else continue the evaluation. 4- Check if the new transaction location falls within the same country. If it does not then flag this transaction as fraud and provide your response else continue the evaluation. 5- If the transaction was not flagged as fraud in the evaluation so far then flag this transaction as genuine and provide your response. Output Format (STRICT)Your response must be structured as follows- Transaction Classification: [Fraud/Genuine]- Reasoning: [Provide a clear explanation following the rules above]- Confidence Score (0-100): [Indicates how likely this transaction is fraudulent based on the criteria]Do not provide vague or inconclusive responses. Follow the rules above with no deviation."

#bad inconclusive prompt
        #user_question = user_question + " Your task is to determine if this transaction appears to be fraud. You will be provided this customers recent genuine payment transactions. You will analyze and understand the spending patterns. Then you will classify the new transaction as fraud or genuine strictly based on the following rules:Rules to Determine Fraud:- Spending Amount Consistency- If the amount of the new transaction is similar to past spending amounts, classify as Genuine.- If the amount is significantly different (e.g., abnormally high or low), classify as Fraud.- Purchase Category Consistency- If the item/service purchased is consistent with past purchases, classify as Genuine.- If the item is entirely different from previous spending habits (e.g., bought a whale fish when past purchases were groceries), classify as Fraud.- Transaction Location Consistency- If the purchase location matches previous transactions, classify as Genuine.- If the purchase occurs in a different country/continent or an #unfamiliar location, classify as Fraud.Output Format (STRICT)Your response must be structured as folo- Transaction Classification: [Fraud/Genuine]- Reasoning: [Provide a clear explanation following the rules above]- Confidence Score (0-100): [Indicates how likely this transaction is fraudulent based on the criteria]Do not provide vague or inconclusive responses. Follow the rules above with no deviation. Flag this transaction as fraud even when only 1 criteria fails."
        #Ollama invoke
        #response = llm.invoke({"question": f"{user_question}\nContext:\n{relevant_text}"})

        #Huggingface invoke
        #user_question = "Customer's new transaction is: " +user_question + "\n Customer's full transactions context is here: \n" + relevant_text
        user_question =  "Customer's new transaction is: " +user_question + "\n Customer's previous genuine spending patterns are here: \n" + relevant_text
        print(f"\nuser_question: {user_question}")
        ai_msg = chat.invoke(user_question)
        extracted_text = ai_msg.content.split("</think>", 1)[-1].strip()
        print(f"\nAI Response:  {extracted_text}")
    else:
        print("Unable to proceed with AI interaction due to missing relevant text.")


# User selection menu
while True:
    choice = input("\nChoose an action:\n1. Insert Text from PDF\n2. Fetch & Interact with ai\n3. Exit\nEnter choice: ")
    if choice == "1":
        insert_text_from_pdf()
    elif choice == "2":
        chat_with_ai()
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")