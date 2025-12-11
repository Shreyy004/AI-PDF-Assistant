import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Now import everything else
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def create_vectorstore(text):
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Use your existing nomic-embed-text model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Use gemma3:1b as LLM
    llm = Ollama(model="gemma3:1b")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain
