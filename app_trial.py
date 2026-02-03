import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from vector_trial import retriever, vector_store

# ------------------- LLM -------------------
from langchain_community.chat_models import ChatHuggingFace
from transformers import pipeline

# HuggingFace pipeline (Flan-T5 Base, lightweight & cloud-friendly)
# Guna task "text2text-generation" dengan Flan-T5
hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

model = ChatHuggingFace(pipeline=hf_pipeline)

# ------------------- Prompt -------------------
prompt = ChatPromptTemplate.from_template("""
You are a Professional HR Policy Assistant supporting employees, managers, and recruiters.

You must determine the user intent FIRST before responding.

--------------------------------
INTENT HANDLING (CRITICAL)
--------------------------------

IF the user message is ONLY a greeting, thanks, or farewell
(e.g. "hello", "hi", "thanks", "thank you", "bye"):

- DO NOT use Summary, Conclusion, Reasoning, or Source.
- Respond with a short, professional greeting.
- Ask the user to ask a question about HR policies with professional.
- STOP. Do not continue.

--------------------------------
POLICY QUESTION HANDLING
--------------------------------

IF the user asks a policy-related question:

- Answer strictly and exclusively using the provided context.
- If the answer is not explicitly stated in the context, respond only with:
  "I don't know."

--------------------------------
RESPONSE RULES (POLICY QUESTIONS ONLY)
--------------------------------

1. Provide a Summary (3–5 sentences).
2. Provide a Conclusion (6–7 sentences).
3. Always include a Source reference (filename + section/heading + page if available).
4. Use neutral, HR-compliant language.
5. Do not infer or assume beyond the context.
6. Keep Summary under 120 words and Conclusion under 200 words.
7. Integrate multiple sections if relevant.
8. Provide a brief justification WITHOUT revealing internal reasoning.

--------------------------------
OUTPUT FORMAT (POLICY QUESTIONS ONLY)
--------------------------------

Justification:
<Which policy sections support the answer>

Summary:
<3–5 sentences>

Conclusion:
<6–7 sentences>

Source:
<filename>, Section <number/heading>, Page <number>

--------------------------------
CONTEXT
--------------------------------
{context}

--------------------------------
QUESTION
--------------------------------
{question}
""")

chain = prompt | model

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖")
st.title("HR Policy Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if question := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)
    if not docs:
        answer = "I don't know"
    else:
        context = "\n\n".join(d.page_content for d in docs[:3])
        response = chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# ------------------- File Upload -------------------
uploaded_files = st.file_uploader(
    "Upload one or more HR brochures (PDF only)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
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
            unique_id = f"{uploaded_file.name}-chunk-{i}-{len(st.session_state.messages)}"
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": uploaded_file.name},
                    id=unique_id
                )
            )

    try:
        vector_store.add_documents(documents, ids=[doc.id for doc in documents])
        vector_store.persist()
        st.success(f"✅ Uploaded and processed {len(uploaded_files)} file(s), {len(documents)} chunks added.")
    except Exception as e:
        st.warning(f"⚠️ Some chunks were skipped due to duplicates.")
