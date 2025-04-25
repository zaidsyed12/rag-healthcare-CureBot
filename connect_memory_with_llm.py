# Phase 2–Connect Memory with LLM
# ● Setup LLM (Mistral with HuggingFace)
# ● Connect LLM with FAISS
# ● Create chain

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step-1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingFace_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingFace_repo_id):
    llm = HuggingFaceEndpoint(repo_id=huggingFace_repo_id, task="text-generation", temperature = 0.7, model_kwargs = {"token":HF_TOKEN, "max_length": "512"} )
    return llm

# Step-2: Connecting LLM with FAISS & Create chain

CUSTOM_PROMPT_TEMPLATE = """

You are a helpful and professional doctor. 
Use the pieces of information provided in the context and your knwoledge to answer user's question.
Your responses should be concise, clear, and compassionate, always aiming to explain medical concepts in a simple way. Keep your responses between 3 to 5 sentences. Answer in a friendly, yet professional tone.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(huggingFace_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
#print("/n/nSOURCE DOCUMENTS: ", response["source_documents"])