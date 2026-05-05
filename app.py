import streamlit as st
import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load the hidden API key
load_dotenv()

# Load the hidden API key
load_dotenv()

# UI Setup
st.set_page_config(page_title="My RAG Tutor", page_icon="📚")
st.title("My Database Systems Tutor")
st.write("Ask me anything about the lecture slides we ingested!")

# Load the Database and LLM (We use caching so it doesn't reload on every single click)
@st.cache_resource
def load_rag_pipeline():
    # Connect to the Chroma database we just built
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
    
    # Set it up as a "retriever" that grabs the top 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Connect to Google Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, transport="rest")

    # Give the AI its strict instructions
    system_prompt = (
        "You are a helpful tutor for a university student studying Database Systems. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep the answer concise and easy to understand.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Wire everything together into a chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Load the brain
chain = load_rag_pipeline()

# The Chat Input Box
user_query = st.text_input("What would you like to know?")

if user_query:
    # Show a loading spinner while Gemini thinks
    with st.spinner("Searching notes and thinking..."):
        print("User asked a question. Converting to math...")
        # Send the question to our chain
        response = chain.invoke({"input": user_query})
        print("Gemini finished thinking! Printing to screen...")
        
        # Print the final answer to the screen
        st.success("Done!")
        st.write(response["answer"])
        
        # Optional: Show exactly which chunks of text the AI used
        with st.expander("Click here to see the exact text chunks the AI used"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content)
                st.divider()