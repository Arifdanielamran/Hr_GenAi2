import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # your PDF vector store

# ------------------- LLM -------------------
model = ChatOllama(model="llama3.2")

# ------------------- Prompt with professional fallback -------------------
prompt = ChatPromptTemplate.from_template("""
You are a professional HR Policy Assistant. 
You must answer questions strictly based on the provided context.

If the answer is not in the context, respond in a professional, polite HR style.
For example, say something like:
"Based on the information provided, I’m unable to determine the answer. 
Please check the relevant HR policy documents or contact HR for clarification."

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

# ------------------- LCEL Chain -------------------
chain = prompt | model

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖")
st.title("HR Policy Chatbot")

# Initialize session state for messages and chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user questions
if question := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ------------------- Retrieve relevant documents -------------------
    docs = retriever.invoke(question)
    if not docs:
        answer = "Based on the information provided, I’m unable to determine the answer. Please check the relevant HR policy documents or contact HR for clarification."
    else:
        context = "\n\n".join(d.page_content for d in docs)[:3000]

        # Run chain
        response = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        answer = response.content

        # Update chat memory
        st.session_state.chat_history.append(("human", question))
        st.session_state.chat_history.append(("assistant", answer))

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Display sources if available
    if docs:
        st.markdown("**Sources:**")
        for d in docs:
            st.markdown(f"- {d.metadata.get('source')}")
