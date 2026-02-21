from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv() 

app = Flask(__name__)


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


embeddings=download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context below:

    Context:
    {context}

    Question:
    {question}
    """
)


retriever = docsearch.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = chain.invoke( msg)
    print("Response : ", response.content)
    return str(response.content)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)