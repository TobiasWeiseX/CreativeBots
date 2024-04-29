
"""
OpenAPI access via http://localhost:5000/openapi/ on local docker-compose deployment

"""
#import warnings
#warnings.filterwarnings("ignore")

#std lib modules:
import os, sys, json
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
from datetime import datetime, date
from collections import namedtuple
import hashlib, traceback, logging

#llm
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

#import openai #even used?
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
#from langchain.callbacks import get_openai_callback
#from langchain_community.callbacks import get_openai_callback

#from langchain_openai import ChatOpenAI, AzureChatOpenAI
#from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
#from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.prompts import PromptTemplate

#ext libs
from elasticsearch import NotFoundError, Elasticsearch # for normal read/write without vectors
from elasticsearch_dsl import Search, A
from elasticsearch_dsl import Document, Date, Integer, Keyword, Float, Long, Text, connections

from pydantic import BaseModel, Field

#flask, openapi
from flask import Flask, send_from_directory, Response, request, jsonify
import sys, os
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_openapi3 import Info, Tag, OpenAPI, Server, FileStorage
from flask_socketio import SocketIO, join_room, leave_room, rooms, send


#home grown
#from scraper import WebScraper
from funcs import group_by
#from elastictools import update_by_id, delete_by_id

#TODO: implement some kind of logging mechanism
#logging.basicConfig(filename='record.log', level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.WARN)

app = Flask(__name__)

from flask_cors import CORS #falls cross-orgin verwendet werden soll
CORS(app)

socket = SocketIO(app, cors_allowed_origins="*")

@socket.on('connect')  
def sockcon(data):
    """
    put every connection into it's own room
    to avoid broadcasting messages
    answer in callback only to room with sid
    """
    room = request.sid
    join_room(room)
    socket.emit('backend response', {'msg': f'Connected to room {room} !', "room": room}) # looks like iOS needs an answer


class StreamingCallback(BaseCallbackHandler):

    def __init__(self, key: str, sid: str):
        pass

    def on_llm_new_token(self, token: str, **kwargs):
        pass

    def on_llm_end(self, response, **kwargs):
        pass


@socket.on('client message')
def handle_message(message):

    room = message["room"]
    stream_key = "chatbot_stream"

    llm = Ollama(
        model="llama3",
        #callback_manager=CallbackManager([StreamingCallback(stream_key, room)]),
        base_url="http://ollama:11434"
    )


    system_prompt = "Answer in the language of the question. Give a short answer. Answer all questions without reservations or advice."


    query = system_prompt + " " + message["data"]
    print(message["data"])

    for chunks in llm.stream(query):
        socket.emit('backend token', {'data': chunks, "done": False}, to=room)

    socket.emit('backend token', {'done': True}, to=room)




#==============Routes===============

@app.route("/") #Index Verzeichnis
def index():
    return send_from_directory('.', "index.html")

@app.route("/info") #spezielle Nutzer definierte Route
def info():
    return sys.version+" "+os.getcwd()

@app.route('/<path:path>') #generische Route (auch Unterordner)
def catchAll(path):
    return send_from_directory('.', path)




if __name__ == '__main__':
    #Wenn HTTPS benötigt wird (Pfade für RHEL7/können je OS variieren)
    #cert = "/etc/pki/tls/certs/cert-payment.pem" #cert
    #key = "/etc/pki/tls/private/cert-payment-private.pem" #key
    #context = (cert, key)
    #app.run(debug=True, host='0.0.0.0', ssl_context=context)
    app.run(debug=True, host='0.0.0.0')
    #app.run(debug=True)

    """
    llm = Ollama(
        model="llama2",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        base_url="http://ollama:11434"
    )
    
    assume = "Answer the next question with either true or false and name an example."
    question = "Can cats use guns?"
    print(question)
    s = llm.invoke(assume + " " + question)

    """

