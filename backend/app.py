
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
from collections import namedtuple
import hashlib, traceback, logging
from functools import wraps
import base64

#-------ext libs--------------
#llm
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import tiktoken
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
#from scraper import WebScraper
from funcs import group_by
from elastictools import get_by_id, update_by_id, delete_by_id
from models import QueryLog, Chatbot, User, Text


#LLM_PAYLOAD = int(os.getenv("LLM_PAYLOAD"))
#CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
BOT_ROOT_PATH = os.getenv("BOT_ROOT_PATH")


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
                #user = get_by_id(client, index="user", id_field_name="email", id_value=data["email"])[0]
                #response = Search(using=client, index="user").filter("term", **{"email": data["email"]})[0:5].execute()
                #response = Search(index="user").filter("term", **{"email": data["email"]})[0:5].execute()
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






def create_key(salt: str, user_email: str) -> Fernet:
    """
    Example salt: 9c46f833b3376c5f3b64d8a93951df4b
    Fernet usage: token = f.encrypt(b"Secret message!")
    """
    salt_bstr = bytes(salt, "utf-8")
    email_bstr = bytes(user_email, "utf-8")
    #password = b"password"
    #salt = os.urandom(16)
    #salt = b"9c46f833b3376c5f3b64d8a93951df4b"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_bstr,
        iterations=48,
    )
    key = base64.urlsafe_b64encode(kdf.derive(email_bstr))
    return Fernet(key)




#app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_METHODS'] = ["GET,POST,OPTIONS,DELETE,PUT"]


env_to_conf = {
    "BACKEND_INTERNAL_URL": "api_url",
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


#TODO add history

def ask_bot(question, bot_id):
    bot = Chatbot.get(id=bot_id)
    llm = Ollama(
        model=bot.llm_model,
        base_url="http://ollama:11434"
    )
    query = bot.system_prompt + " " + question
    for chunk in llm.stream(query):
        yield chunk





#from flask_cors import CORS #falls cross-orgin verwendet werden soll
#CORS(app)

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



def hash_password(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


#======================= TAGS =============================

jwt_tag = Tag(name='JWT', description='Requires a valid JSON Web Token')
not_implemented_tag = Tag(name='Not implemented', description='Functionality not yet implemented beyond an empty response')
debug_tag = Tag(name='Debug', description='Debug')


bot_tag = Tag(name='Bot', description='Bot')


#==============Routes===============

class LoginRequest(BaseModel):
    email: str = Field(None, description='The users E-Mail that serves as nick too.')
    password: str = Field(None, description='A short text by the user explaining the rating.')


@app.post('/login', summary="", tags=[], security=security)
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


#-----bot routes------

from speech import text_to_speech


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
    return ""


class UpdateBotRequest(BaseModel):
    id: str = Field(None, description="The bot's id")

@app.put('/bot', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def update_bot(form: UpdateBotRequest, decoded_jwt, user):
    """
    Changes a chatbot
    """

    return ""




class AskBotRequest(BaseModel):
    bot_id: str = Field(None, description="The bot's id")
    question: str = Field(None, description="The question the bot should answer")



@app.get('/bot/ask', summary="", tags=[bot_tag], security=security)
@uses_jwt()
def query_bot(query: AskBotRequest, decoded_jwt, user):
    """
    Asks a chatbot
    """
    start = datetime.now().timestamp()

    bot_id = query.bot_id
    prompt = query.question


    history = ""

    system_prompt = "Antworte freundlich, mit einer ausführlichen Erklärung, sofern vorhanden auf Basis der folgenden Informationen. Please answer in the language of the question."


    prompt_template = system_prompt +"""
    <ctx>    
        {context}
    </ctx>
    <hs>
   """+ history +"""
    </hs>    
    Question: {question}
    """

    chat_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
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
    #query = bot.system_prompt + " " + question
    #for chunk in llm.stream(query):
    #    yield chunk


    #chunk_size = 1536
    #chunk_overlap = 200
    LLM_PAYLOAD=16384
    CHUNK_SIZE=1536


    k = int(LLM_PAYLOAD / CHUNK_SIZE) - 1
    if (k < 2):
        k = 2

    #scoredocs = vector_store.similarity_search_with_score(prompt, k=k+10)
    scoredocs = vector_store.similarity_search_with_score(prompt, k=k+10)


    query = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=False,
        return_source_documents=True,
        retriever=vector_store.as_retriever(search_kwargs={'k': k}),
        chain_type_kwargs={"prompt": chat_prompt}
    )

    #inputTokens = 0
    #outputTokens = 0

    #with get_openai_callback() as cb:
    qares = query.invoke({'query': prompt})
    qadocs = qares['source_documents'] # STS: deliver doc names and page numbers in the future

    ls = [dict(x) for x in qadocs]

    for qadoc in qadocs:
        print(qadoc, flush=True)



    for x in scoredocs:
        #xs = [x.to_dict() for x in scoredocs]
        print(x, flush=True)


    r = qares['result']




    #r = ""
    #for chunk in ask_bot(question=query.question, bot_id=query.bot_id):
    #    r += chunk




    duration = round(datetime.now().timestamp() - start, 2)

    app.logger.info(duration)

    return jsonify({
        "answer": r,
        "duration": str(duration),
        "docs": ls#,
        #"score_docs": xs
    })




#-----------------Embedding----------------------
ESDocument = namedtuple('Document', ['page_content', 'metadata'])

class TrainTextRequest(BaseModel):
    bot_id: str = Field(None, description="The bot's id")
    text: str = Field(None, description="Some text")

#TODO: needs to be reimplemented with another mechanism like celeery to manage longer running tasks and give feedback to frontend

@app.post('/bot/train/text', summary="", tags=[jwt_tag], security=security)
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

    embeddings = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")

    vector_store = ElasticsearchStore(
        es_url=app.config['elastic_uri'],
        index_name= "chatbot_" + bot_id.lower(),
        embedding=embeddings
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return jsonify({
        "status": "success"
    })










#======== DEBUG routes ============

@app.get('/debug/schema', summary="", tags=[debug_tag])
def get_schema():
    """

    """
    #chatbots = query.chatbots
    #client = Elasticsearch(app.config['elastic_uri'])

    def simplify_properties(d):
        new_d = {}
        for field, d3 in d["properties"].items():
            if "type" in d3:
                new_d[field] = d3["type"]
            elif "properties" in d3:
                new_d[field] = simplify_properties(d3)
        return new_d


    def get_type_schema(client: Elasticsearch):
        d = client.indices.get(index="*").body
        new_d = {}
        for index, d2 in d.items():
            new_d[index] = simplify_properties(d2["mappings"])
        return new_d

    return jsonify( get_type_schema(client) )


#TODO: route that takes a schema json and compares to internal structure and returns boolean


#-------- non api routes -------------

@app.route("/") #Index Verzeichnis
def index():
    return send_from_directory('./public', "index.html")


@app.route('/<path:path>') #generische Route (auch Unterordner)
def catchAll(path):
    #return send_from_directory('.', path)
    return send_from_directory('./public', path)



def init_indicies():
    # create the mappings in elasticsearch
    for Index in [QueryLog, Chatbot, User, Text]:
        Index.init()


def create_default_users():
    #create default users
    client = Elasticsearch(app.config['elastic_uri'])
    default_users = os.getenv("DEFAULT_USERS")
    if default_users:
        for (email, pwd, role) in json.loads(default_users):
            if len(get_by_id(client, index="user", id_field_name="email", id_value=email)) == 0:
                user = User(email=email, password_hash=hash_password(pwd + email), role=role)
                user.creation_date = datetime.now()
                user.save()




if __name__ == '__main__':

    #TODO: implement some kind of logging mechanism
    #logging.basicConfig(filename='record.log', level=logging.DEBUG)
    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.WARN)

    """
    USE_LOKI_LOGGER = os.getenv("USE_LOKI_LOGGER")
    if USE_LOKI_LOGGER:
        handler = logging_loki.LokiHandler(
            url="http://loki:3100/loki/api/v1/push", 
            tags={"application": "Nextsearch"},
            #auth=("username", "password"),
            version="1",
        )
        app.logger.addHandler(handler)
    """

    connections.create_connection(hosts=app.config['elastic_uri'])

    #client = Elasticsearch(app.config['elastic_uri'])
    #client = Elasticsearch(hosts=[{"host": "elasticsearch"}], retry_on_timeout=True)
    client = Elasticsearch(app.config['elastic_uri'], retry_on_timeout=True)


    #TODO: find a clean way to wait without exceptions!
    #Wait for elasticsearch to start up!
    i = 1
    while True:
        try:
            #client = Elasticsearch(app.config['elastic_uri'])

            client.cluster.health(wait_for_status='yellow')
            print("Elasticsearch found! Run Flask-app!", flush=True)
            break
        except ConnectionError:
            i *= 1.5
            time.sleep(i)
            print("Elasticsearch not found! Wait %s seconds!" % i, flush=True)


    # Display cluster health
    #app.logger.debug(connections.get_connection().cluster.health())


    init_indicies()
    create_default_users()
    app.run(debug=True, threaded=True, host='0.0.0.0')






