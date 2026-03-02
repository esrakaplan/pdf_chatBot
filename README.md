#PDF -> Text Splitter -> Embedding -> Vector DB -> LLM

text splitter (chunking) : langchain_text_splitters import RecursiveCharacterTextSplitter
Embedding : HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorDb : FAISS.from_documents(docs, embeddings)
llm = Ollama(model="llama2")


# to do
1)
pip install pypdf
python -m pip install langchain langchain-community langchain-openai faiss-cpu langchain-text-splitters


2)
ollama download & install

ollama pull llama2




OPEN AI Solutions
#api_key=os.getenv("OPENAI_API_KEY")
#embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
#llm = ChatOpenAI(openai_api_key=api_key, temperature=0)


embedding options :
1)
from langchain_community.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

2)
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")



