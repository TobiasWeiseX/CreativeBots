
"""
OpenAPI access via http://localhost:5000/openapi/ on local docker-compose deployment

"""
#import warnings
#warnings.filterwarnings("ignore")

#------std lib modules:-------
import os, sys, json, time
import os.path
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
from datetime import datetime, date
#from collections import namedtuple
import hashlib, traceback, logging
from functools import wraps
import base64

#-------ext libs--------------
#llm
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.prompts import PromptTemplate

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings

#from langchain_community.vectorstores.elasticsearch import ElasticsearchStore  #deprecated
from langchain_elasticsearch import ElasticsearchStore
from uuid import uuid4

from elasticsearch import NotFoundError, Elasticsearch # for normal read/write without vectors
from elasticsearch_dsl import Search, A, Document, Date, Integer, Keyword, Float, Long, Text, connections
from elasticsearch.exceptions import ConnectionError

from pydantic import BaseModel, Field

import logging_loki
import jwt as pyjwt


#flask, openapi
from flask import Flask, send_from_directory, send_file, Response, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_openapi3 import Info, Tag, OpenAPI, Server, FileStorage
from flask_socketio import SocketIO, join_room, leave_room, rooms, send

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

#----------home grown--------------
from lib.funcs import group_by
from lib.elastictools import get_by_id, update_by_id, delete_by_id, wait_for_elasticsearch
from lib.models import init_indicies, QueryLog, Chatbot, User, Text
from lib.chatbot import ask_bot, train_text, download_llm
from lib.speech import text_to_speech
from lib.mail import send_mail
from lib.user import hash_password, create_user, create_default_users


BOT_ROOT_PATH = os.getenv("BOT_ROOT_PATH")
assert BOT_ROOT_PATH


# JWT Bearer Sample
jwt = {
    "type": "http",
    "scheme": "bearer",
    "bearerFormat": "JWT"
}
security_schemes = {"jwt": jwt}
security = [{"jwt": []}]


info = Info(
    title="Chatbot-API",
    version="1.0.0",
    summary="The REST-API",
    description="Default model: ..."
)
servers = [
    Server(url=BOT_ROOT_PATH )
]

class NotFoundResponse(BaseModel):
    code: int = Field(-1, description="Status Code")
    message: str = Field("Resource not found!", description="Exception Information")

app = OpenAPI(
        __name__,
        info=info,
        servers=servers,
        responses={404: NotFoundResponse},
        security_schemes=security_schemes
    )


def uses_jwt(required=True):
    """
    Wraps routes in a jwt-required logic and passes decoded jwt and user from elasticsearch to the route as keyword
    """

    def non_param_deco(f):
        @wraps(f)
        def decorated_route(*args, **kwargs):
            token = None
            if "Authorization" in request.headers:
                token = request.headers["Authorization"].split(" ")[1]


            if not token:
                if required:
                    return jsonify({
                        'status': 'error',
                        "message": "Authentication Token is missing!",
                    }), 401

                else:
                    kwargs["decoded_jwt"] = {}
                    kwargs["user"] = None
                    return f(*args, **kwargs)


            try:
                data = pyjwt.decode(token, app.config["jwt_secret"], algorithms=["HS256"])
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    "message": "JWT-decryption: " + str(e)
                }), 401

            try:
                response = User.search().filter("term", **{"email": data["email"]})[0:5].execute()
                for hit in response:
                    user = hit
                    break

            except Exception as e:
                return jsonify({
                    'status': 'error',
                    "message": "Invalid Authentication token!"
                }), 401

            kwargs["decoded_jwt"] = data
            kwargs["user"] = user
            return f(*args, **kwargs)

        return decorated_route

    return non_param_deco




env_to_conf = {
    "ELASTIC_URI": "elastic_uri",
    "SECRET": "jwt_secret"
}

#import values from env into flask config and do existence check
for env_key, conf_key in env_to_conf.items():
    x = os.getenv(env_key)
    if not x:
        msg = "Environment variable '%s' not set!" % env_key
        app.logger.fatal(msg)
        sys.exit(1)
    else:
        app.config[conf_key] = x




socket = SocketIO(app, cors_allowed_origins="*")

@socket.on('connect')  
def sockcon(data):
    """
    put every connection into it's own room
    to avoid broadcasting messages
    answer in callback only to room with sid
    """
    room = request.sid + request.remote_addr
    join_room(room)
    socket.emit('backend response', {'msg': f'Connected to room {room} !', "room": room}) # looks like iOS needs an answer


#TODO: pydantic message type validation


@socket.on('client message')
def handle_message(message):

    #try:
    room = message["room"]
    question = message["question"]
    system_prompt = message["system_prompt"]
    bot_id = message["bot_id"]

    #except:
    #    return

    for chunk in ask_bot(system_prompt + " " + question, bot_id):
        socket.emit('backend token', {'data': chunk, "done": False}, to=room)
    socket.emit('backend token', {'done': True}, to=room)


#======================= TAGS =============================

not_implemented_tag = Tag(name='Not implemented', description='Functionality not yet implemented beyond an empty response')
debug_tag = Tag(name='Debug', description='Debug')
bot_tag = Tag(name='Bot', description='Bot')
user_tag = Tag(name='User', description='User')

#==============Routes===============

class LoginRequest(BaseModel):
    email: str = Field(None, description='The users E-Mail that serves as nick too.')
    password: str = Field(None, description='A short text by the user explaining the rating.')


@app.post('/user/login', summary="", tags=[user_tag])
def login(form: LoginRequest):
    """
    Get your JWT to verify access rights
    """

    if form.email is None or form.password is None:
        msg = "Invalid password!"
        app.logger.error(msg)
        return jsonify({
            'status': 'error',
            'message': msg
        }), 400

    client = Elasticsearch(app.config['elastic_uri'])
    match get_by_id(client, index="user", id_field_name="email", id_value=form.email):
        case []:
            msg = "User with email '%s' doesn't exist!" % form.email
            app.logger.error(msg)
            return jsonify({
                'status': 'error',
                'message': msg
            }), 400

        case [user]:
            if user["password_hash"] == hash_password(form.password + form.email):
                token = pyjwt.encode({"email": form.email}, app.config['jwt_secret'], algorithm="HS256")
                #app.logger.info(token)
                return jsonify({
                    'status': 'success',
                    'jwt': token
                })
            else:
                msg = "Invalid password!"
                app.logger.error(msg)
                return jsonify({
                    'status': 'error',
                    'message': msg
                }), 400


class RegisterRequest(BaseModel):
    email: str = Field(None, description='The users E-Mail that serves as nick too.')
    password: str = Field(None, description='A short text by the user explaining the rating.')


@app.post('/user/register', summary="", tags=[user_tag])
def register(form: RegisterRequest):
    """
    Register an account
    """

    if form.email is None or form.password is None:
        msg = "Parameters missing!"
        app.logger.error(msg)
        return jsonify({
            'status': 'error',
            'message': msg
        }), 400


    if User.get(id=form.email, ignore=404) is not None:
        return jsonify({
            'status': 'error',
            "message": "User with that e-mail address already exists!"
        })

    else:
        user = User(meta={'id': form.email})
        user.creation_date = datetime.now()
        user.email = form.email
        user.password_hash = hash_password(form.password + form.email)
        user.role = "User"
        user.isEmailVerified = False
        user.save()

        msg = """
        <h1>Verify E-Mail</h1>

        Hi!

        Please click on the following link to verify your e-mail:


        <a href="http://127.0.0.1:5000/">Click here!</a>

        """

        send_mail(user.email, "User registration @ Creative Bots", "Creative Bots", msg)

        return jsonify({
            'status': 'success'
        })



#-----bot routes------


class GetSpeechRequest(BaseModel):
    text: str = Field(None, description="Some text to convert to mp3")

@app.post('/text2speech', summary="", tags=[], security=security)
def text2speech(form: GetSpeechRequest):
    file_name = text_to_speech(form.text)

    #return send_file(file_path, mimetype='audio/mpeg') #, attachment_filename= 'Audiofiles.zip', as_attachment = True)
    return jsonify({
        "status": "success",
        "file": "/" + file_name
    })



class GetBotRequest(BaseModel):
    id: str = Field(None, description="The bot's id")

@app.get('/bot', summary="", tags=[bot_tag], security=security)
@uses_jwt(required=False)
def get_bots(query: GetBotRequest, decoded_jwt, user):
    """
    List all bots or one by id
    """
    match query.id:
        case None:
            match user:
                case None:
                    #get all public bots
                    ls = []
                    for hit in Chatbot.search()[0:10000].execute():
                        d = hit.to_dict()
                        if d["visibility"] == "public":
                            d["id"] = hit.meta.id
                            ls.append(d)

                    return jsonify(ls)

                case _:
                    #get all user bots
                    ls = []
                    for hit in Chatbot.search()[0:10000].execute():
                        d = hit.to_dict()
                        if "creator_id" in d:
                            if user.meta.id == d["creator_id"]:
                                d["id"] = hit.meta.id
                                ls.append(d)

                    return jsonify(ls)

        case some_id:
            match user:
                case None:
                    bot = Chatbot.get(id=query.id)
                    if bot.visibility == "public":
                        d = bot.to_dict()
                        d["id"] = bot.meta.id
                        return jsonify(d)
                    else:
                        return jsonify(None)
                case _:
                    bot = Chatbot.get(id=query.id)
                    d = bot.to_dict()
                    d["id"] = bot.meta.id
                    return jsonify(d)






class CreateBotRequest(BaseModel):
    name: str = Field(None, description="The bot's name")
    visibility: str = Field('private', description="The bot's visibility to other users ('private', 'public')")
    description: str = Field('', description="The bot's description of purpose and being")
    system_prompt: str = Field('', description="The bot's defining system prompt")
    llm_model: str = Field("llama3", description="The bot's used LLM")

    #status = Keyword()
    #temperature = Float()


@app.post('/bot', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def create_bot(form: CreateBotRequest, decoded_jwt, user):
    """
    Creates a chatbot for the JWT associated user.
    """
    bot = Chatbot()
    bot.name = form.name
    bot.visibility = form.visibility
    bot.description = form.description
    bot.system_prompt = form.system_prompt
    bot.llm_model = form.llm_model

    #add meta data
    bot.creation_date = datetime.now()
    bot.creator_id = user.meta.id
    bot.save()

    return jsonify({
        "bot_id": bot.meta.id
    })



class DeleteBotRequest(BaseModel):
    id: str = Field(None, description="The bot's id")

@app.delete('/bot', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def delete_bot(form: DeleteBotRequest, decoded_jwt, user):
    """
    Deletes a chatbot via it's id
    """
    bot = Chatbot.get(id=form.id)
    bot.delete()

    return jsonify({
        "status": "success"
    })


class UpdateBotRequest(BaseModel):
    id: str = Field(None, description="The bot's id")

@app.put('/bot', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def update_bot(form: UpdateBotRequest, decoded_jwt, user):
    """
    Changes a chatbot
    """

    return jsonify({
        "status": "success"
    })



class AskBotRequest(BaseModel):
    bot_id: str = Field(None, description="The bot's id")
    question: str = Field(None, description="The question the bot should answer")




from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



@app.get('/bot/ask', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def query_bot(query: AskBotRequest, decoded_jwt, user):
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

    ollama_url = os.getenv("OLLAMA_URI")


    embeddings = OllamaEmbeddings(model="llama3", base_url=ollama_url)

    vector_store = ElasticsearchStore(
            es_url=app.config['elastic_uri'],
            index_name= "chatbot_" + bot_id.lower(),
            distance_strategy="COSINE",
            embedding=embeddings
         )


    bot = Chatbot.get(id=bot_id)
    llm = Ollama(
        model=bot.llm_model,
        base_url=ollama_url
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

    app.logger.info(duration)

    return jsonify({
        "answer": r,
        "duration": str(duration),
        #"docs": ls#,
        "score_docs": xs
    })




#-----------------Embedding----------------------

class TrainTextRequest(BaseModel):
    bot_id: str = Field(None, description="The bot's id")
    text: str = Field(None, description="Some text")


@app.post('/bot/train/text', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def upload(form: TrainTextRequest, decoded_jwt, user):
    """
    Caution: Long running request!
    """
    bot_id = form.bot_id
    text = form.text

    # validate body
    if not bot_id:
        return jsonify({
            'status': 'error',
            'message': 'chatbotId is required'
        }), 400

    if not text:
        return jsonify({
            'status': 'error',
            'message': 'No data source found'
        }), 400

    train_text(bot_id, text)
    return jsonify({
        "status": "success"
    })

#-------- non api routes -------------

@app.route("/") #Index Verzeichnis
def index():
    return send_from_directory('./public', "index.html")


@app.route('/<path:path>') #generische Route (auch Unterordner)
def catchAll(path):
    #return send_from_directory('.', path)
    return send_from_directory('./public', path)



def main():
    LOG_LEVEL = os.getenv("LOG_LEVEL")
    if LOG_LEVEL:
        logging.basicConfig(level=eval("logging." + LOG_LEVEL))
    else:
        logging.basicConfig(level=logging.WARN)

    #TODO: implement some kind of logging mechanism

    """
    USE_LOKI_LOGGER = os.getenv("USE_LOKI_LOGGER")
    if USE_LOKI_LOGGER:
        handler = logging_loki.LokiHandler(
            url="http://loki:3100/loki/api/v1/push", 
            tags={"application": "CreativeBots"},
            #auth=("username", "password"),
            version="1",
        )
        app.logger.addHandler(handler)
    """

    #wait_for_elasticsearch()
    download_llm()
    connections.create_connection(hosts=app.config['elastic_uri'], request_timeout=60)
    wait_for_elasticsearch()
    init_indicies()
    create_default_users()
    app.run(debug=False, threaded=True, host='0.0.0.0')


if __name__ == '__main__':
    main()





