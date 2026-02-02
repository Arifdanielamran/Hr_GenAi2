import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector import retriever

# ------------------- LLM -------------------
model = ChatOllama(model="llama3.2")

# ------------------- Prompt -------------------
prompt = ChatPromptTemplate.from_template("""
You are a professional HR Policy Assistant. 
You must answer questions strictly based on the provided context. 
If the answer is not in the context, respond only with: "I don't know".

Always follow these rules:
1. Provide a **medium-length Summary** (3–5 sentences, clear and concise).  
2. Include a **Source reference** (filename and section number/heading if available).  
3. Use professional, neutral, HR‑compliant language.  
4. If multiple sections are relevant, summarize them together but keep the answer short and structured.  
5. Never invent or assume information outside the context.  
6. If the user greets (e.g., "hello", "hi"), respond politely with a short greeting and invite them to ask about HR policies.  
7. If the user says "thank you", respond with a polite closing like "You're welcome. Let me know if you need help with HR policies."  
8. If the user says "goodbye", respond with "Goodbye. Wishing you a smooth day at work."  

Format your answer exactly like this:
**Summary:** <medium-length conclusion>  
*Source:* <filename>, Section <number or heading>

Context:
{context}

Question:
{question}
""")

# ------------------- Memory + Chain -------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    combine_docs_chain=prompt | model
)

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

    # Run chain with memory
    response = chain.invoke({"question": question})
    answer = response["answer"] if isinstance(response, dict) else str(response)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)