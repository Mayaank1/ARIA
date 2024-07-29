
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import List
from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import boto3
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


mentalassistant="""
    Role:Your role is that of  AI assistant for mental health.\
    The user will ask a question for help regarding mental health or as an answer to a question based on chat history.\
    Your goal is to help users using cognitive behavioral therapy by giving a understanding,empathetic,engaging and discursive response\
    . You should be knowledgeable about all aspects of this technique\
    and be able to provide clear and concise answers to users’ questions.\
    {context}
Use the following chat history provided in brackets aid in the answer to the question\
    {chat_history}
 Question: {question} 
"""
PROMPT = PromptTemplate(template=mentalassistant, input_variables=["context", "chat_history", "question"])

bedrock = boto3.client(service_name="bedrock-runtime")
modelPath = "sentence-transformers/all-MiniLM-l6-v2"


model_kwargs = {'device':'cpu'}

encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs 
    )
faiss_index = FAISS.load_local("db",embeddings,allow_dangerous_deserialization=True)
retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':50})
memory = ConversationSummaryBufferMemory(llm=llm,memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=40) 
#print(memory)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': PROMPT}, 
    verbose=True,

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

    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':50})
    return llm

def get_response_llm(llm, vectorstore_faiss, query):    
    result = qa.invoke({"question": query})
    retrieved_docs = retriever.invoke(query)
    #print(retrieved_docs)
    result['answer'] = re.split('answer:',result['answer'])[-1]
    print(f"{result['chat_history']}\n")
    return result['answer']


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
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            
            st.success("Done")
    

if __name__ == "__main__":
    main()
