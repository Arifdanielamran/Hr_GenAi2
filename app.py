import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# ------------------- LLM -------------------
model = ChatOllama(model="llama3.2")

# ------------------- Prompt -------------------
prompt = ChatPromptTemplate.from_template("""
You are a professional HR Policy Assistant. 
You must answer questions strictly based on the provided context. 
If the answer is not in the context, respond only with: "I don't know".

Always follow these rules:
1. Provide a clear, concise **Summary** (no verbatim copy).
2. Include a **Source reference** (filename and section number/heading if available).
3. Use professional, neutral, HR‑compliant language.
4. If multiple sections are relevant, summarize them together but keep the answer short and structured.
5. Never invent or assume information outside the context.

Format your answer exactly like this:
**Summary:** <short, clear conclusion>  
*Source:* <filename>, Section <number or heading>

Context:
{context}

Question:
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