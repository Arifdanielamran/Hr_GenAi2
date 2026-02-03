import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# ------------------- LLM -------------------
model = ChatOllama(model="llama3.2")

# ------------------- Prompt -------------------
prompt = ChatPromptTemplate.from_template("""
You are a professional HR Policy Assistant. 
Your role is to answer employee and recruiter questions strictly based on the provided context. 
If the answer is not in the context, respond only with: "I don't know".

Follow these rules:
1. Provide a **medium-length explanation** (3–5 sentences) that is clear, professional, and concise.  
2. Add a **Conclusion** (6–7 sentences) that synthesizes the key points into a final, actionable statement.  
3. Always include a **Source reference** (filename and section number/heading if available).  
4. Use neutral, HR‑compliant language suitable for policy documentation.  
5. If multiple sections are relevant, summarize them together in a structured way.  
6. Never invent or assume information outside the context.  
7. Respond politely to greetings, thanks, or farewells with short, professional phrases.  
8. **Explain your reasoning step‑by‑step before writing the Summary and Conclusion.**  
9. **Keep the Summary under 120 words and the Conclusion under 200 words.**  
10. **Use professional but approachable language, as if speaking to HR managers and employees.**

Format your answer exactly like this:
**Reasoning:** <step‑by‑step explanation of how you derived the answer>  
**Summary:** <3–5 sentence explanation>  
**Conclusion:** <6–7 sentence final statement>  
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