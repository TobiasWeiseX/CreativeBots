
__version__ = "0.4.0"

#------std lib modules:-------
import os, sys, json, time
import os.path
from typing import Any, Tuple, List, Dict, Any, Callable, Optional, Union
from datetime import datetime, date
from functools import wraps

#-------ext libs--------------
from elasticsearch_dsl import connections

from pydantic import BaseModel, Field
import jwt as pyjwt

#flask, openapi
from flask import Flask, send_from_directory, send_file, Response, request, jsonify
from flask_openapi3 import Info, Tag, OpenAPI, Server #FileStorage
from flask_socketio import SocketIO, join_room, leave_room, rooms, send
from werkzeug.utils import secure_filename

import asyncio
import logging

#----------home grown--------------
from lib.funcs import group_by
from lib.elastictools import get_by_id, wait_for_elasticsearch
from lib.models import init_indicies, Chatbot, User, Text, Question, Answer
from lib.chatbot import ask_bot, train_text, download_llm
from lib.speech import text_to_speech
from lib.mail import send_mail
from lib.user import hash_password, create_user, create_default_users
from lib.logging import ElasticsearchLogHandler, get_log_level


BOT_ROOT_PATH = os.getenv("BOT_ROOT_PATH")
assert BOT_ROOT_PATH

ollama_url = os.getenv("OLLAMA_URI")
assert ollama_url

elastic_url = os.getenv("ELASTIC_URI")
assert elastic_url

jwt_secret = os.getenv("SECRET")
assert jwt_secret



class MeasureTime:
    
    def __init__(self, name, logger=logging.getLogger()):
        self.__start = datetime.now().timestamp()
        self.__logger = logger
        self.__name = name

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        duration = round(datetime.now().timestamp() - self.__start, 2)
        self.__logger.info(f"{self.__name} duration: {duration} seconds")
        print(f"{self.__name} duration: {duration} seconds", flush=True)
        return False



def uses_jwt(logger=None, required=True):
    """
    Wraps routes in a jwt-required logic and passes decoded jwt and user from elasticsearch to the route as keyword
    """

    jwt_secret = os.getenv("SECRET")
    assert jwt_secret

    def non_param_deco(f):
        @wraps(f)
        def decorated_route(*args, **kwargs):

            #app.logger.info(f"Request from IP: '{request.remote_addr}'")

            #request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

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
                data = pyjwt.decode(token, jwt_secret, algorithms=["HS256"])
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





def create_app():

    # JWT Bearer Sample
    jwt = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    }
    security_schemes = {"jwt": jwt}
    security = [{"jwt": []}]


    info = Info(
        title="CreativeBots-API",
        version=__version__,
        summary="The REST-API to manage bots, users and more!",
        description=""
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

    logger = logging.getLogger()

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


    class SocketMessage(BaseModel):
        question: str = Field(None, description="Status Code")
        system_prompt: str = Field(None, description="Status Code")
        bot_id: str = Field(None, description="Status Code")
        room: Union[str, None] # = Field(None, description="Status Code")


    #TODO: pydantic message type validation

    @socket.on('client message')
    def handle_message(message):

        SocketMessage.model_validate(message)

        logger.info("Starting stream")


        #try:
        room = message["room"]
        question = message["question"]
        system_prompt = message["system_prompt"]
        bot_id = message["bot_id"]


        q = Question()
        q.text = question
        q.md5 = hash_password(question)
        q.save()


        #start = datetime.now().timestamp()

        with MeasureTime("Streaming + Scoring docs -total", logger) as timer:

            d = ask_bot(system_prompt + " " + question, bot_id)

            def get_scores(*args):
                score_docs = d["get_score_docs"]()
                return score_docs


            def do_streaming(*args):
                start_stream = datetime.now().timestamp()
                answer = ""
                for chunk in d["answer_generator"]():
                    socket.emit('backend token', {'data': chunk, "done": False}, to=room)
                    answer += chunk

                stream_duration = round(datetime.now().timestamp() - start_stream, 2)

                logger.info("Stream duration: " + str(stream_duration))

                a = Answer(
                        question_id=q.meta.id,
                        answer=answer,
                        md5=hash_password(answer)
                    )
                a.save()



            async def f():
                ls = await asyncio.gather(
                    asyncio.to_thread(get_scores, 1,2,3),
                    asyncio.to_thread(do_streaming, 1,2,3)
                )
                return ls

            [score_docs, _] = asyncio.run(f())

            socket.emit('backend token', {
                'done': True,
                "score_docs": score_docs
            }, to=room)



        #duration = round(datetime.now().timestamp() - start, 2)
        #logger.info("Total duration: " + str(duration))




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
        LoginRequest.model_validate(form)


        if form.email is None or form.password is None:
            msg = "Invalid password!"
            logger.error(msg)
            return jsonify({
                'status': 'error',
                'message': msg
            }), 400

        match get_by_id(index="user", id_field_name="email", id_value=form.email):
            case []:
                msg = "User with email '%s' doesn't exist!" % form.email
                logger.error(msg)
                return jsonify({
                    'status': 'error',
                    'message': msg
                }), 400

            case [user]:
                if user["password_hash"] == hash_password(form.password + form.email):
                    token = pyjwt.encode({"email": form.email}, jwt_secret, algorithm="HS256")
                    #logger.info(token)
                    return jsonify({
                        'status': 'success',
                        'jwt': token
                    })
                else:
                    msg = "Invalid password!"
                    logger.error(msg)
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
            logger.error(msg)
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

        with MeasureTime("text2speech file", logger) as timer:
            file_name = text_to_speech(form.text)

        logger.info("Created '%s' !" % file_name)

        return jsonify({
            "status": "success",
            "file": "/" + file_name
        })



    #============ Bot CRUD ===============

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
        CreateBotRequest.model_validate(form)

        bot = Chatbot(
            name=form.name,
            visibility=form.visibility,
            description=form.description,
            system_prompt=form.system_prompt,
            llm_model=form.llm_model
        )

        #add meta data
        bot.creation_date = datetime.now()
        bot.creator_id = user.meta.id
        bot.save()

        return jsonify({
            "bot_id": bot.meta.id
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




    class UpdateBotRequest(BaseModel):
        id: str = Field(None, description="The bot's id")
        name: str = Field(None, description="The bot's name")
        visibility: str = Field(None, description="The bot's visibility to other users ('private', 'public')")
        description: str = Field(None, description="The bot's description of purpose and being")
        system_prompt: str = Field(None, description="The bot's defining system prompt")
        llm_model: str = Field(None, description="The bot's used LLM")


    @app.put('/bot', summary="", tags=[bot_tag], security=security)
    @uses_jwt()
    def update_bot(form: UpdateBotRequest, decoded_jwt, user):
        """
        Change a chatbot via it's id
        """
        bot = Chatbot.get(id=form.id)

        bot.name = form.name
        bot.visibility = form.visibility
        bot.description = form.description
        bot.system_prompt = form.system_prompt
        bot.llm_model = form.llm_model

        #add meta data
        bot.changed_date = datetime.now()
        bot.save()

        return jsonify({
            "status": "success"
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



    #============================================================================


    class AskBotRequest(BaseModel):
        bot_id: str = Field(None, description="The bot's id")
        question: str = Field(None, description="The question the bot should answer")


    @app.get('/bot/ask', summary="", tags=[bot_tag], security=security)
    @uses_jwt()
    def query_bot(query: AskBotRequest, decoded_jwt, user):
        """
        Asks a chatbot
        """

        bot_id = query.bot_id
        question = query.question

        start = datetime.now().timestamp()

        d = ask_bot(system_prompt + " " + question, bot_id)

        def get_scores(*args):
            score_docs = d["get_score_docs"]()
            return score_docs


        def do_streaming(*args):
            start_stream = datetime.now().timestamp()
            answer = ""
            for chunk in d["answer_generator"]():
                answer += chunk

            stream_duration = round(datetime.now().timestamp() - start_stream, 2)
            logger.info(f"Stream duration: {stream_duration}")
            return answer


        async def f():
            ls = await asyncio.gather(
                asyncio.to_thread(get_scores, 1,2,3),
                asyncio.to_thread(do_streaming, 1,2,3)
            )
            return ls

        [score_docs, answer] = asyncio.run(f())

        duration = round(datetime.now().timestamp() - start, 2)

        logger.info("Total duration: " + str(duration))

        return jsonify({
            "answer": answer,
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


        with MeasureTime("Training on text", logger) as _:
            train_text(bot_id, text)

        return jsonify({
            "status": "success"
        })

    #-------- non api routes -------------

    @app.route("/") #Index Verzeichnis
    def index():
        return send_from_directory('./public', "index.html")


    @app.route("/info")
    def debug_info():
        import multiprocessing

        def get_module_versions():
            with open("requirements.txt", "r") as f:
                modules = {line.split("#")[0] for line in f.read().split("\n") if line.split("#")[0] != ""}
            with open("current_requirements.txt", "r") as f:
                d = {k: v for (k, v) in [line.split("#")[0].split("==") for line in f.read().split("\n") if line.split("#")[0] != ""]}
            return {k: v for k, v in d.items() if k in modules}

        d = {}
        d["module_versions"] = get_module_versions()
        d["OLLAMA_NUM_PARALLEL"] = os.getenv("OLLAMA_NUM_PARALLEL")
        d["OLLAMA_MAX_LOADED_MODELS"] = os.getenv("OLLAMA_MAX_LOADED_MODELS")
        d["cpus"] = multiprocessing.cpu_count()

        #return "CPUs: " + str(cpus) + "<br>" + json.dumps(get_module_versions(), indent=4).replace("\n", "<br>")

        return json.dumps(d, indent=4).replace("\n", "<br>")




    @app.route('/<path:path>') #generische Route (auch Unterordner)
    def catchAll(path):
        logger.info(f"Path requested: '{path}'")
        return send_from_directory('./public', path)



    lvl = get_log_level()
    #print(f"Log level: {lvl}", flush=True)

    logger.addHandler(ElasticsearchLogHandler(level=logging.INFO))
    logger.setLevel(lvl)

    download_llm("llama3")
    connections.create_connection(hosts=elastic_url, request_timeout=60)
    wait_for_elasticsearch()
    init_indicies()
    create_default_users()

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0')


