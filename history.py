import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models.bedrock import BedrockChat
import boto3
import re

def get_pdf_text():
    loader = PyPDFDirectoryLoader("data")
    doc = loader.load()
    return doc


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
    chunk_overlap= 500,
    separators=["\n\n","\n"," ",""])
    text = text_splitter.split_documents(documents= text)
    return text


def get_vectorstore(text_chunks, query):
    model_kwargs = {'device':'cpu'}

    encode_kwargs = {'normalize_embeddings': False}
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs 
    )
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    #Prompttemplate

    prompt_template = """You are a helpful, empathetic, and insightful AI conversationalist. Your primary goal is to engage in a meaningful conversation with the human user, listen to their concerns, and gently guide the discussion to uncover any underlying issues they may be experiencing. Use your responses to build rapport, show empathy, and provide thoughtful insights or solutions.
    Answer  the questions only if you know the answer. otherwise say I dont know 
Use the following  context provided in brackets aid in the answer to the question\
    context:{context}
 Question: {question} 
    
    
    keep the answers precise to the question
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context" ,"question"]
    )
    chain_type_kwargs = { "prompt" : PROMPT }

    #LLM
    bedrock = boto3.client(service_name="bedrock-runtime")
    llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock)
    memory = ConversationBufferMemory(memory_key='chat_history',output_key='answer', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs=chain_type_kwargs
    )

    response  = conversation_chain({"question": query})
    print(response.get("chat_history"))
    response['answer'] = re.split('Answer:',response['answer'])[-1]
    return(response.get("answer"))

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat Assistant")

    # get pdf text
    raw_text = get_pdf_text()

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    user_question =  st.chat_input("Ask a Question")


    if "messages" not in st.session_state.keys():
        st.session_state["messages"] = [{"role": "assistant",
                                         "content": "Hello there, how can i help you"}]

    if "messages" in st.session_state.keys():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if user_question is not None:
        st.session_state.messages.append({
            "role":"user",
            "content":user_question
        })

        with st.chat_message("user"):
            st.write(user_question)


    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading"):
                output = get_vectorstore(text_chunks, user_question)
                ai_response = output
                st.write(ai_response)

        new_ai_message = {"role":"assistant","content": ai_response}
        st.session_state.messages.append(new_ai_message)


if __name__ == '__main__':
    main()