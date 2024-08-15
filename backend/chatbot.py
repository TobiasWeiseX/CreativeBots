
from models import Chatbot, Text, User

from uuid import uuid4
from collections import namedtuple
import os, hashlib, traceback, logging
from datetime import datetime, date

#from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections
from elasticsearch_dsl import connections

#from langchain.callbacks.manager import CallbackManager
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.chains import RetrievalQA
#from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
#from langchain.prompts import PromptTemplate

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings

from langchain_elasticsearch import ElasticsearchStore


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


ollama_url = os.getenv("OLLAMA_URI")
elastic_url = os.getenv("ELASTIC_URI")

assert ollama_url
assert elastic_url

ESDocument = namedtuple('Document', ['page_content', 'metadata'])

#TODO: needs to be reimplemented with another mechanism like celeery to manage longer running tasks and give feedback to frontend

def train_text(bot_id, text):
    """
    Caution: Long running request!
    """

    bot = Chatbot.get(id=bot_id)
    user = User.get(id=bot.creator_id)

    t = Text()
    t.text = text
    t.md5 = hashlib.md5(text.encode()).hexdigest()

    #add meta data
    t.creation_date = datetime.now()
    t.creator_id = user.meta.id
    t.save()

    #train with given text
    chunk_size = 1536
    chunk_overlap = 200

    documents = []
    for i, s in enumerate(RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len).split_text(text)):
        documents.append(ESDocument(
            page_content=s,
            metadata={
                "segment_nr": i,
                "text_id": t.meta.id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        ))

    embeddings = OllamaEmbeddings(model=bot.llm_model, base_url=ollama_url)

    vector_store = ElasticsearchStore(
        es_url=elastic_url,
        index_name= "chatbot_" + bot_id.lower(),
        embedding=embeddings
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return True



#TODO add history

def ask_bot(question, bot_id):
    bot = Chatbot.get(id=bot_id)
    llm = Ollama(
        model=bot.llm_model,
        base_url=ollama_url
    )
    query = bot.system_prompt + " " + question
    for chunk in llm.stream(query):
        yield chunk



#connections.get_connection()

#if es.indices.exists(index="index"):




def ask_bot2(question, bot_id):
    bot = Chatbot.get(id=bot_id)
    llm = Ollama(
        model=bot.llm_model,
        base_url=ollama_url
    )
    query = bot.system_prompt + " " + question
    for chunk in llm.stream(query):
        yield chunk



    #def query_bot(query: AskBotRequest, decoded_jwt, user):
    """
    Asks a chatbot
    """
    start = datetime.now().timestamp()

    bot_id = query.bot_id
    prompt = query.question


    system_prompt = (
        "Antworte freundlich, mit einer ausführlichen Erklärung, sofern vorhanden auf Basis der folgenden Informationen. Please answer in the language of the question."
        "\n\n"
        "{context}"
    )


    ch_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    embeddings = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")

    vector_store = ElasticsearchStore(
            es_url=app.config['elastic_uri'],
            index_name= "chatbot_" + bot_id.lower(),
            distance_strategy="COSINE",
            embedding=embeddings
         )


    bot = Chatbot.get(id=bot_id)
    llm = Ollama(
        model=bot.llm_model,
        base_url="http://ollama:11434"
    )

    k = 4
    scoredocs = vector_store.similarity_search_with_score(prompt, k=k)

    retriever = vector_store.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, ch_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    r = ""
    #for chunk in rag_chain.stream({"input": "What is Task Decomposition?"}):
    for chunk in rag_chain.stream({"input": prompt}):
        print(chunk, flush=True)
        if "answer" in chunk:
            r += chunk["answer"]



    #for chunk in ask_bot(question=query.question, bot_id=query.bot_id):
    #    r += chunk



    xs = []
    for doc, score in scoredocs:
        #print(doc.__dict__, flush=True)
        #print(doc, flush=True)
        xs.append([dict(doc), score])


    duration = round(datetime.now().timestamp() - start, 2)

    #app.logger.info(duration)

    #return jsonify({
    #    "answer": r,
    #    "duration": str(duration),
    #    #"docs": ls#,
    #    "score_docs": xs
    #})












