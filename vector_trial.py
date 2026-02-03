import os
import glob
import shutil
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings   # ✅ Import HuggingFace embeddings

# ---------- PDF folder ----------
pdf_folder = r"C:\Users\syeda\Documents\RagMiniProject\Hr_GenAi2\hr_brochures"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

documents = []

for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={"source": os.path.basename(pdf_file)},
                id=f"{os.path.basename(pdf_file)}-chunk-{i}"
            )
        )

# ---------- EMBEDDINGS ----------
# Replace OllamaEmbeddings with HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"   # ✅ You can change to any HF model
)

# ---------- VECTOR STORE ----------
db_location = "./chroma_langchain_db"

# Always rebuild DB if you want fresh embeddings
if os.path.exists(db_location):
    shutil.rmtree(db_location)

vector_store = Chroma(
    collection_name="hr_brochures",
    embedding_function=embeddings,
    persist_directory=db_location
)

if len(documents) == 0:
    print("⚠️ No documents found in hr_brochures/")
else:
    vector_store.add_documents(documents, ids=[doc.id for doc in documents])
    print(f"✅ Added {len(documents)} document chunks to the vector store.")

# ---------- RETRIEVER ----------
retriever = vector_store.as_retriever(search_kwargs={"k": 5})