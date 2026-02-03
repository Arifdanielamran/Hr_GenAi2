from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from vector import retriever

# ------------------- LLM -------------------
llm = ChatOllama(model="llama3.2")

# ------------------- Memory -------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ------------------- Prompt -------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional HR Policy Assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ------------------- RAG Chain -------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

# ------------------- Usage -------------------
response = qa_chain.run("Explain grievance reporting policy")
print(response)
