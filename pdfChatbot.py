import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama


pdf_path = r"test.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# --- chunking ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

print(f"Total chunk count: {len(docs)}")

# --- 3. Embeddings (HuggingFace CPU model) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- 4. FAISS vector db ---
db = FAISS.from_documents(docs, embeddings)

# --- Most related 3 chunks ---
query = "What is the document about ?"
relevant_docs = db.similarity_search(query, k=3)

llm = Ollama(model="llama2")
context = "\n".join([doc.page_content for doc in relevant_docs])
prompt = f"PDF Context:\n{context}\n\nQuery: {query}"

response = llm.invoke(prompt)
print("Answer:\n", response)