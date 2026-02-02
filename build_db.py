import os, glob, shutil
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- PDF folder ----------
pdf_folder = r"C:\Users\syeda\Documents\RagMiniProject\RagEnv\hr_brochures"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

documents = []
for pdf_file in pdf_files:
    
    reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"source": os.path.basename(pdf_file)},
            id=f"{os.path.basename(pdf_file)}-chunk-{i}"
        ))

# ---------- EMBEDDINGS ----------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ---------- VECTOR STORE ----------
db_location = "./chroma_langchain_db"
shutil.rmtree(db_location, ignore_errors=True)  # delete lama bila rebuild

vector_store = Chroma(
    collection_name="hr_brochures",
    embedding_function=embeddings,
    persist_directory=db_location
)

vector_store.add_documents(documents, ids=[doc.id for doc in documents])
vector_store.persist()
print(f"✅ Added {len(documents)} chunks to the vector store.")