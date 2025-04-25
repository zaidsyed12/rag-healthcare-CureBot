from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Setup of LLM 
def load_llm():
    return ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo"  
    )

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and professional doctor. 
Use the pieces of information provided in the context and your knowledge to answer user's question.
Your responses should be concise, clear, and compassionate, always aiming to explain medical concepts in a simple way. Keep your responses between 3 to 5 sentences. Answer in a friendly, yet professional tone.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# loading Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Building Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT:", response["result"])
# print("Source Documents:", response["source_documents"])