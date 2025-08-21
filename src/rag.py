from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
from globals import configs
import os

# Load documents from PDFs in the specified directory
pdf_loader = DirectoryLoader(
    path = configs["rag"]["database_path"], 
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)
documents = []
documents.extend(pdf_loader.load())

# Split documents into smaller chunks for processing
chunks = RecursiveCharacterTextSplitter(
    chunk_size = configs["rag"]["chunk_size"],
    chunk_overlap = configs["rag"]["chunk_overlap"],
    length_function = len,
    is_separator_regex = False
    ).split_documents(documents)

# Define the embedding model
_ = load_dotenv(configs["server"]["env_file_path"])
embeddings = GoogleGenerativeAIEmbeddings(
    model = configs["rag"]["embedding_model"]["name"],
    google_api_key = os.environ.get(configs["chat"]["llm_model"]["api_env_var_name"]),
    task_type = configs["rag"]["embedding_model"]["task_type"]
    )

# Define the language model
llm = ChatGoogleGenerativeAI(
    model = configs["chat"]["llm_model"]["name"],
    google_api_key = os.environ.get(configs["chat"]["llm_model"]["api_env_var_name"]),
    temperature = configs["chat"]["llm_model"]["temperature"],
    max_output_tokens = configs["chat"]["llm_model"]["max_tokens"],
    top_k = configs["chat"]["llm_model"]["top_k"],
    top_p = configs["chat"]["llm_model"]["top_p"]
    )

# Create a vector store from the document chunks using the embeddings
vector_store = Chroma.from_documents(documents = chunks, embedding = embeddings)

# Define the prompt template for the RetrievalQA chain
template = f'{configs["chat"]["system_prompt"]}' + """

Context:
{context}

Question:
{question}

Answer:
"""

# Create the RetrievalQA chain
chat_qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vector_store.as_retriever(
        search_type = configs["rag"]["retriever"]["search_type"], 
        search_kwargs = configs["rag"]["retriever"]["search_kwargs"]
        ),
    return_source_documents = True,
    chain_type_kwargs = {"prompt": PromptTemplate.from_template(template)}
)

# Function to invoke the chat QA system
def invoke_qa(prompt: str):
    """
    Invoke the chat QA system with the provided prompt.
    
    Args:
        prompt (str): A dictionary containing the query.
        
    Returns:
        str: The result of the chat QA invocation.
    """
    response = chat_qa.invoke({"query": prompt})
    return response["result"]
