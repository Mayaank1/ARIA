'''from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3"
) 

output=llm.invoke("Tell me a joke")
print(output)
'''
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import List
from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

prompt_template = """
H: Use the following pieces of context to provide a
concise answer to the question at the end but use atleast summarize with
250 words with detailed explaantions. If you don't know the answer,
just say that you don't know, don't try to make up an answer.

<context>
{context}
</context

Question: {question}
A:"""



PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


modelPath = "sentence-transformers/all-MiniLM-l6-v2"


model_kwargs = {'device':'cpu'}

encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs 
    )

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    
    return docs

def get_vector_store(docs):
    
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("db")
    return db

def get_llama3_llm():

    llm = Ollama(temperature =0,model="llama3")
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs)
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chatbot")

    user_question = st.text_input("Ask a Question ")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")


    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            llm=get_llama3_llm()
            faiss_index = FAISS.load_local("db",embeddings,allow_dangerous_deserialization=True)
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
