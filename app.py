import streamlit as st
import os

from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings




if "vector" not in st.session_state:
    # st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.embeddings =HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    st.session_state.loader=PyPDFLoader('C:\\Users\\hp\\Downloads\\MNITSIP2441.pdf')
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatBot Demo")
llm=Ollama(model="llama2")

# prompt = ChatPromptTemplate.from_template(
#     """
#     <s>[INST] <<SYS>>
#         Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question  
#     <</SYS>>

#     <context>
#         { context }
#     </context>

#     Questions:{ input }
#         [/INST]

#     """
# )
template="""
        Answer the questions based on the provided context only. 
        context:{context}
        Please provide the most accurate response based on the question in a very concise and to the point manner.  
        question:{input}
        """
    
prompt=PromptTemplate( input_variables=['context','input'],template=template)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input you prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
