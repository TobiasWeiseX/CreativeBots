"""
All functions around bots

"""
from uuid import uuid4
from collections import namedtuple
import os, hashlib, traceback, logging
from datetime import datetime, date
from elasticsearch_dsl import connections

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings

from langchain_elasticsearch import ElasticsearchStore

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from lib.models import Chatbot, Text, User


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
    txt_md5 = hashlib.md5(text.encode()).hexdigest()
    t = Text.get(id=txt_md5, ignore=404)
    if t is not None:
        return True

    else:
        bot = Chatbot.get(id=bot_id)
        user = User.get(id=bot.creator_id)

        t = Text(meta={'id': txt_md5})
        t = Text()
        t.text = text
        t.md5 = txt_md5

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
    """
    Asks a chatbot
    """

    bot = Chatbot.get(id=bot_id)
    prompt = question
    system_prompt = bot.system_prompt + "\n\n{context}"

    rag_index = "chatbot_" + bot_id.lower()

    if connections.get_connection().indices.exists(index=rag_index):

        vector_store = ElasticsearchStore(
            es_url=elastic_url,
            index_name=rag_index,
            distance_strategy="COSINE",
            embedding=OllamaEmbeddings(model=bot.llm_model, base_url=ollama_url)
        )

        def gen_func():
            llm = Ollama(
                model=bot.llm_model,
                base_url=ollama_url
            )

            ch_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            retriever = vector_store.as_retriever()
            question_answer_chain = create_stuff_documents_chain(llm, ch_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            for chunk in rag_chain.stream({"input": prompt}):
                print(chunk, flush=True)
                if "answer" in chunk:
                    yield chunk["answer"]


        def get_score_docs():
            k = 4
            start_vec_search = datetime.now().timestamp()
            scoredocs = vector_store.similarity_search_with_score(prompt, k=k)
            vec_search_duration = round(datetime.now().timestamp() - start_vec_search, 2)
            print("Vec search duration: ", vec_search_duration, flush=True)
            xs = []
            for doc, score in scoredocs:
                #print(doc.__dict__, flush=True)
                #print(doc, flush=True)
                xs.append([score, dict(doc)])
            return xs


        return {
            "answer_generator": gen_func,
            "get_score_docs": get_score_docs
        }

    else:

        def gen_func():
            bot = Chatbot.get(id=bot_id)
            llm = Ollama(
                model=bot.llm_model,
                base_url=ollama_url
            )
            query = bot.system_prompt + " " + question
            for chunk in llm.stream(query):
                yield chunk

        return {
            "answer_generator": gen_func,
            "get_score_docs": lambda: []
        }




from ollama import Client as OllamaClient

def download_llm(model):
    #print(ollama_url, flush=True)

    #ollama_client = OllamaClient(host=ollama_url)
    #x = ollama_client.pull('llama3')
    #print( type(x), flush=True)
    #print( x.__dict__, flush=True)
    #print( x, flush=True)

    s = """curl %s/api/pull -d '{ "name": "%s" }' """ % (ollama_url, model)
    print( os.system(s.strip()) ,flush=True)

