

"""
OpenAPI access via http://localhost:5000/openapi/ on local docker-compose deployment
"""

#------std lib modules:-------
import os, sys, json, time
import os.path
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
from datetime import datetime, date
import logging
from functools import wraps

#-------ext libs--------------

from elasticsearch_dsl import connections

from pydantic import BaseModel, Field
import jwt as pyjwt

import asyncio

#----------home grown--------------
from lib.funcs import group_by
from lib.elastictools import get_by_id, wait_for_elasticsearch
from lib.models import init_indicies, Chatbot, User, Text
from lib.chatbot import ask_bot, ask_bot2, train_text, download_llm
from lib.speech import text_to_speech
from lib.mail import send_mail
from lib.user import hash_password, create_user, create_default_users

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi_socketio import SocketManager

from jinja2 import Environment, FileSystemLoader




BOT_ROOT_PATH = os.getenv("BOT_ROOT_PATH")
assert BOT_ROOT_PATH

ollama_url = os.getenv("OLLAMA_URI")
assert ollama_url

elastic_url = os.getenv("ELASTIC_URI")
assert elastic_url

jwt_secret = os.getenv("SECRET")
assert jwt_secret



app = FastAPI()
socket_manager = SocketManager(app=app)



@app.sio.on('connect')
async def sockcon(sid, data):
    """
    put every connection into it's own room
    to avoid broadcasting messages
    answer in callback only to room with sid
    """
    room = request.sid + request.remote_addr
    join_room(room)
    await app.sio.emit('backend response', {'msg': f'Connected to room {room} !', "room": room}) # looks like iOS needs an answer



@app.sio.on('client message')
async def handle_message(sid, message):

    #try:
    room = message["room"]
    question = message["question"]
    system_prompt = message["system_prompt"]
    bot_id = message["bot_id"]

    start = datetime.now().timestamp()
    d = ask_bot2(system_prompt + " " + question, bot_id)

    def get_scores(*args):
        score_docs = d["get_score_docs"]()
        return score_docs


    def do_streaming(*args):
        start_stream = datetime.now().timestamp()
        for chunk in d["answer_generator"]():
            socket.emit('backend token', {'data': chunk, "done": False}, to=room)
        stream_duration = round(datetime.now().timestamp() - start_stream, 2)
        print("Stream duration: ", stream_duration, flush=True)



    [score_docs, _] = await asyncio.gather(
        asyncio.to_thread(get_scores, 1,2,3),
        asyncio.to_thread(do_streaming, 1,2,3)
    )


    await app.sio.emit.emit('backend token', {
        'done': True,
        "score_docs": score_docs
    }, to=room)

    duration = round(datetime.now().timestamp() - start, 2)
    print("Total duration: ", duration, flush=True)





#@app.sio.on('join')
#async def handle_join(sid, *args, **kwargs):
#    await app.sio.emit('lobby', 'User joined')


#@sm.on('leave')
#async def handle_leave(sid, *args, **kwargs):
#    await sm.emit('lobby', 'User left')









class JobSearch(BaseModel):
    location: str
    language: str


@app.post("/search")
def job_search(js: JobSearch):

    #https://berlinstartupjobs.com/?s=python&page=3

    location = "Berlin"
    radius = 50




@app.get("/")
async def root():
    template = env.get_template('index.twig')
    html = template.render()
    return HTMLResponse(html)



