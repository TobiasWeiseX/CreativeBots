"""
OpenAPI access via http://localhost/epdm/chat/bot/openapi/ on local docker-compose deployment

"""
import warnings
warnings.filterwarnings("ignore")

#std lib modules:
import os, sys, json
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
from datetime import datetime, date
from collections import namedtuple
import hashlib, traceback, logging

#ext libs:
import requests
import chardet
import codecs

from elasticsearch import NotFoundError, Elasticsearch # for normal read/write without vectors
from elasticsearch_dsl import Search, A
from elasticsearch_dsl import Document, Date, Integer, Keyword, Float, Long, Text, connections


import openai #even used?
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore

#from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.prompts import PromptTemplate


from pydantic import BaseModel, Field

#flask, openapi
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_openapi3 import Info, Tag, OpenAPI, Server, FileStorage
from flask_socketio import SocketIO, join_room, leave_room, rooms

#home grown
from scraper import WebScraper
from funcs import group_by
from elastictools import update_by_id, delete_by_id

#TODO: implement some kind of logging mechanism
#logging.basicConfig(filename='record.log', level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.WARN)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING")
OPENAI_TEXT_EMBEDDING = os.getenv("OPENAI_TEXT_EMBEDDING")
LLM_PAYLOAD = int(os.getenv("LLM_PAYLOAD"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
BOT_ROOT_PATH = os.getenv("BOT_ROOT_PATH")



# required settings

assert OPENAI_API_KEY
assert BOT_ROOT_PATH

from models import NextsearchLog

info = Info(title="Bot-API", version="1.0.0",
            summary="",
            description="chatGPT model: " + OPENAI_MODEL_NAME)
servers = [
    Server(url=BOT_ROOT_PATH ) #+ '/')
]
app = OpenAPI(__name__, info=info, servers=servers)

# init app and env
#app = Flask(__name__)
index_prefix = "chatbot"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_METHODS'] = ["GET,POST,OPTIONS,DELETE,PUT"]

# set cors
CORS(app)

env_to_conf = {
    "BACKEND_INTERNAL_URL": "api_url",
    "ELASTIC_URI": "elastic_uri"
}

#import values from env into flask config and do existence check
for env_key, conf_key in env_to_conf.items():
    x = os.getenv(env_key)
    if not x:
        msg = "Environment variable '%s' not set!" % env_key
        app.logger.fatal(msg)
        #raise Exception(msg)
        sys.exit(1)
    else:
        app.config[conf_key] = x



if OPENAI_API_TYPE == 'azure':
    EMBEDDING = AzureOpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, 
        deployment=OPENAI_EMBEDDING, 
        model=OPENAI_TEXT_EMBEDDING, 
        openai_api_type=OPENAI_API_TYPE, 
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        chunk_size = CHUNK_SIZE
    ) 
else:
    EMBEDDING = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)



# socket = SocketIO(app, cors_allowed_origins="*", path="/epdm/chat/bot/socket.io")
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
    socket.emit('my response', {'data': 'Connected'}) # looks like iOS needs an answer

class StreamingCallback(BaseCallbackHandler):
    # def __init__(self, key, sourceFileId):
    def __init__(self, key: str, sid: str):
        # print("StreamingCallback")
        self.key = key
        self.sid = sid
        self.text = ""
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs):
        # print("on_llm_new_token", token)
        self.text += token
        self.new_sentence += token
        socket.emit(self.key, self.text, to=self.sid)

    def on_llm_end(self, response, **kwargs):
        # print("on_llm_end", response)
        self.text = ""
        socket.emit(self.key, "[END]", to=self.sid)


#-----------------

class RatingRequest(BaseModel):
    queryId: str = Field(None, description='The query-id. Example: fa9d2f024698b723931fe633bfe065d3')
    rating: int = Field(0, ge=-1, le=1, description='A rating value of: -1, 0 or 1 for bad, neutral or good')
    reason: str = Field("", description='A short text by the user explaining the rating.')
    reasonTags: List[str] = Field([], description='A list of tags flagging the rating. Examples: ["LAZY", "CLEVER", "COOL", "VILE", "BUGGED", "WRONG"]') #limit tag len to 64 chars

@app.post('/bot/rating', summary="Allows for answers to be rated", tags=[])
def rating(body: RatingRequest):
    """
    Gives a rating to an answer.
    """
    queryId = body.queryId
    rating = body.rating
    reason = body.reason
    reasonTags = body.reasonTags

    client = Elasticsearch(app.config['elastic_uri'])

    try:
        update_by_id(client, index="nextsearch_log", id_field_name="queryid", id_value=queryId, values_to_set={
            "rating": rating,
            "reason": reason,
            "reasonTags": reasonTags
        })
    except NotFoundError:
        return jsonify({
            'status': 'error',
            'message': "Unknown id: '%s!'" % queryId
        }), 404

    return "", 201


#--------------

def get_slugs_for_names(client: Elasticsearch):
    s = Search(using=client, index="chatbot")
    s = s[0:10000]
    response = s.execute()
    return { d["slug"]: d["chatbotName"] for d in (x.to_dict() for x in response.hits)}


#DEAD?
class BarChartRequest(BaseModel):
    chatbots: List[str] = Field([], description="""A list of chatbot names to filter for""")
    start: datetime = Field("2000-01-31T16:47+00:00", description="""The interval start datetime in <a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a> format""")
    end: datetime = Field("2100-01-31T16:47+00:00", description="""The interval end datetime in <a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a> format""")

@app.get('/bot/usage/activityPerDay', summary="", tags=[])
def usage_activity_per_day(query: BarChartRequest):
    """

    """
    chatbots = query.chatbots
    start = query.start
    end = query.end


    client = Elasticsearch(app.config['elastic_uri'])

    id2name = get_slugs_for_names(client)

    s = Search(using=client, index="nextsearch_log") \
        .filter("range", timest={"gte": start}) \
        .filter("range", timest={"lte": end})
    
    s = s[0:10000] #if not used size is set to 10 results

    def maybe_id2name(id):
        if id in id2name:
            return id2name[id]
        return id

    def agg_pretty_result(d):
        ls = []
        for bucket in d["aggregations"]["day"]["buckets"]:
            d = {}
            d["date"] = bucket["key_as_string"]
            d["bots"] = []
            for bot in bucket["chatbot"]["buckets"]:
                d["bots"].append({
                    "bot": maybe_id2name(bot["key"]),
                    "cost": bot["cost_per_bot"]["value"]
                })
            ls.append(d)
        return ls


    s.aggs.bucket('day', 'terms', field='date')
    s.aggs['day'].bucket('chatbot', 'terms', field='chatbotid').metric('cost_per_bot', 'sum', field='inCt')

    response = s.execute()

    r = response.to_dict()
    del r["hits"]

    ls = agg_pretty_result(r)

    return jsonify(ls)

#--------------


#book_tag = Tag(name="book", description="Some Book")

class DonutChartRequest(BaseModel):
    chatbots: List[str] = Field([], description="""A list of chatbot names to filter for""")
    start: datetime = Field("2000-01-31T16:47+00:00", description="""The interval start datetime in <a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a> format""")
    end: datetime = Field("2100-01-31T16:47+00:00", description="""The interval end datetime in <a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a> format""")

@app.get('/bot/usage/activity', summary="Takes an interval and gives back a summary of bots and their activity and cost.", tags=[])
def usage_activity(query: DonutChartRequest):
    """
    Use datetime in ISO 8601 format: 2007-08-31T16:47+00:00
    """
    chatbots = query.chatbots
    start = query.start
    end = query.end

    #group nextsearch_log by chatbotid and sum inCt

    client = Elasticsearch(app.config['elastic_uri'])

    id2name = get_slugs_for_names(client)

    s = Search(using=client, index="nextsearch_log") \
        .filter("range", timest={"gte": start}) \
        .filter("range", timest={"lte": end})
    
    s = s[0:10000] #if not used size is set to 10 results

    a = A('terms', field='chatbotid') \
        .metric('cost_per_bot', 'sum', field='inCt')

    s.aggs.bucket('bots', a)
    response = s.execute()
    #print(response.aggregations.bots.buckets)

    def maybe_id2name(id):
        if id in id2name:
            return id2name[id]
        return id

    match chatbots:
        case []:
            #ls = [d for d in (d.to_dict() for d in response.aggregations.bots.buckets)]
            ls = [{**d, "chatbotname": maybe_id2name(d["key"].split("_")[0])} for d in (d.to_dict() for d in response.aggregations.bots.buckets)]
        case _:
            ls = [{**d, "chatbotname": maybe_id2name(d["key"].split("_")[0])} for d in (d.to_dict() for d in response.aggregations.bots.buckets) if d["key"] in id2name and id2name[d["key"]] in chatbots]

    d = {
        "chart": {
            "series": {
                "data": ls
            }
        }
    }

    return jsonify(d)
    #return jsonify(ls)
    #return jsonify(list(response.aggregations.bots.buckets))

#------------------

class RetrieveChatRequest(BaseModel):
    sessionId: str = Field(None, description="""The session's id. Example: d73bccba29b6376c1869944f26c3b670""")

@app.get('/bot/usage/conversation', summary="Takes a session-id and gives you all of it's content.", tags=[])
def usage_conversation(query: RetrieveChatRequest):
    """
    Example session-id: d73bccba29b6376c1869944f26c3b670
    """
    sessionId = query.sessionId

    client = Elasticsearch(app.config['elastic_uri'])

    s = Search(using=client, index="nextsearch_log") \
        .filter("term", session=sessionId)

    s = s[0:10000] #if not used size is set to 10 results
    response = s.execute()
    return jsonify([hit.to_dict() for hit in response])

#------------

class DialogTableRequest(BaseModel):
    chatbots: List[str] = Field([], description="""A list of chatbot names to filter for""")
    start: datetime = Field("2000-01-31T16:47+00:00", description="""The interval start datetime in <a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a> format""")
    end: datetime = Field("2100-01-31T16:47+00:00", description="""The interval end datetime in <a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a> format""")

@app.get('/bot/usage/conversations', summary="Takes an interval and gives you all chatbots and their sessions within.", tags=[])
def usage_conversations(query: DialogTableRequest):
    """
    Use datetime in ISO 8601 format: 2007-08-31T16:47+00:00
    """
    #GET /bot/usage/conversations?chatbots=robobot,cyberbot,gigabot&timeStart=2024-01-15&timeEnd=2024-01-15&timeStart=00:00&timeEnd=23:00
    chatbots = query.chatbots
    start = query.start
    end = query.end

    client = Elasticsearch(app.config['elastic_uri'])

    s = Search(using=client, index="nextsearch_log") \
        .filter("range", timest={"gte": start}) \
        .filter("range", timest={"lte": end})
    
    s = s[0:10000] #if not used size is set to 10 results

    #a = A('terms', field='chatbotid') \
    #    .metric('cost_per_bot', 'sum', field='inCt')
    #s.aggs.bucket('bots', a)

    response = s.execute()
    hits = (x.to_dict() for x in response.hits)

    id2name = get_slugs_for_names(client)

    match chatbots:
        case []:
            pass
        case _:
            
            hits = filter(lambda d: (id2name[d["chatbotid"]] in chatbots) if d["chatbotid"] in id2name else False, hits)


    d = group_by([lambda d: d["chatbotid"], lambda d: d["session"] ], hits)

    d2 = {}
    for chatbotid, v in d.items():
        if chatbotid in id2name:
            d2[id2name[chatbotid]] = v

    return jsonify(d2)


#------------------

class ExtractUrlRequest(BaseModel):
    url: str = Field(None, description="""The URL to a website whose HTML-embedded URLs you'd like to have.""", strict=True)

@app.post('/bot/extract-url', summary="Get URLs from a website via its URL", tags=[])
def extract_url(body: ExtractUrlRequest):
    """
    Takes a json of form {"url": "..."} and gives back a list of URLs found within the specified URL's HTML-sourcecode.
    """
    url = body.url
    if not url:
        return jsonify({'status': 'error', 'message': 'Missing required parameter url!'}), 400

    with WebScraper() as web_scraper:
        return jsonify(web_scraper.extract_urls(url))

#------------------

def extract_data(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Webscrape pages of the given links and return a list of texts
    """
    with WebScraper() as web_scraper:
        return web_scraper.extract_page_data(links)


def get_word_splits(word_file: str) -> List:
    loader = Docx2txtLoader(word_file)
    pages = loader.load_and_split()
    txt_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    doc_list = []
    for page in pages:
        pg_splits = txt_spliter.split_text(page.page_content)
        doc_list.extend(pg_splits)
    return doc_list


def get_text_splits_from_file(text_file: str) -> List:
    # Detect the file's encoding
    with open(text_file, 'rb') as file:
        encoding_result = chardet.detect(file.read())

    # Use the detected encoding to read the file
    detected_encoding = encoding_result['encoding']
    with codecs.open(text_file, 'r', encoding=detected_encoding, errors='replace') as file:
        text = file.read()

    return get_text_splits(text)


def determine_index(chatbot_id: str) -> str:
    return f"{index_prefix}_{chatbot_id.lower()}"


def embed_index(doc_list: List[Dict[str, str]], chatbot_id: str) -> None:
    """
    Add source documents in chatbot_xyz index!
    """
    index = determine_index(chatbot_id)

    #print(f"add documents to index {index}", flush=True)
    app.logger.info(f"add documents to index {index}")


    #ElasticsearchStore.from_documents(doc_list, EMBEDDING, index_name=index, es_url=elastic_uri)
    ElasticsearchStore.from_documents(doc_list, EMBEDDING, index_name=index, es_url=app.config['elastic_uri'])






class TrainForm(BaseModel):
    #url: str = Field(None, description="""""", strict=True)
    chatbotSlug: str = Field(None, description="""""")
    files: List[FileStorage] = Field(None, description="""Some files""")
    text: str = Field(None, description="Some text")
    #filesMetadata: List[Dict[str, str]] = Field(None, description="""""")

    filesMetadata: str = Field(None, description="""A JSON string""") #a json string: [ ... ]
    links: str  = Field(None, description="""A JSON string""") #a json? [ ... ]


#TODO: needs to be reimplemented with another mechanism like celeery to manage longer running tasks and give feedback to frontend
@app.post('/bot/train', summary="", tags=[])
def upload(form: TrainForm):
    """
    Caution: Long running request!
    """
    #url = body.url

    #print(form.file.filename)
    #print(form.file_type)
    #form.file.save('test.jpg')

    #app.logger.info("TRAIN called!")

    # extract body
    chatbot_id = form.chatbotSlug
    files = form.files
    text = form.text
    files_metadata = form.filesMetadata
    if files_metadata:
        files_metadata = json.loads(files_metadata)
    links = form.links
    if links:
        links = json.loads(links) #[{url: '...'}] ?
        app.logger.debug(links)


    # validate body
    if not chatbot_id:
        return jsonify({
            'status': 'error',
            'message': 'chatbotId is required'
        }), 400

    if not files_metadata and not text and not links:
        return jsonify({
            'status': 'error',
            'message': 'No data source found'
        }), 400

    if files_metadata and len(files) != len(files_metadata):
        return jsonify({
            'status': 'error',
            'message': 'Number of uploaded files metadata and files should be same'
        }), 400

    if links and len(links) == 0:
        return jsonify({
            'status': 'error',
            'message': 'No links found'
        }), 400



    try:

        # store raw data and extract doc_list
        os.makedirs(f"{app.config['UPLOAD_FOLDER']}/{chatbot_id}", exist_ok=True)

        #train with given files
        for i, file in enumerate(files):
            filename = files_metadata[i]["slug"] + "_" + secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], chatbot_id, filename)
            file.save(file_path)

            app.logger.info("File saved successfully!")

            doc_list = []
            match file.filename.split(".")[-1]:
                case "pdf":
                    doc_list = get_pdf_splits(file_path)
                    doc_list = add_metadata(
                                    doc_list=doc_list,
                                    source_type="pdf_file",
                                    chatbot_id=chatbot_id,
                                    source_file_id=files_metadata[i]["slug"],
                                    filename=file.filename
                                )
                case "txt":
                    doc_list = get_text_splits_from_file(file_path)
                    doc_list = add_metadata(
                                    doc_list=doc_list,
                                    source_type="text_file",
                                    chatbot_id=chatbot_id,
                                    source_file_id=files_metadata[i]["slug"],
                                    filename=file.filename
                                )
                case "docx" | "doc":
                    doc_list = get_word_splits(file_path)
                    doc_list = add_metadata(
                                    doc_list=doc_list,
                                    source_type="word_file",
                                    chatbot_id=chatbot_id,
                                    source_file_id=files_metadata[i]["slug"],
                                    filename=file.filename
                                )
                case _:
                    app.logger.error("Unknown file extension: '%s'!" % file.filename.split(".")[-1])


            # embed file doc_list
            embed_index(doc_list=doc_list, chatbot_id=chatbot_id)

        #train with given text
        if text:
            doc_list = get_text_splits(text)
            doc_list = add_metadata(
                            doc_list=doc_list,
                            source_type="text",
                            chatbot_id=chatbot_id,
                            source_file_id="text",
                            txt_id=hashlib.md5(text.encode()).hexdigest()
                        )

            # embed raw text doc_list
            embed_index(doc_list=doc_list, chatbot_id=chatbot_id)

        #train with given links
        if links and len(links) > 0:
            links_docs = extract_data(links)
            for i, doc in enumerate(links_docs):
                if not doc['text']:
                    app.logger.info(f"Document {i} '{doc['url']} of {len(links_docs)} doesn't contain text. Skip.")

                else:
                    app.logger.info(f"embed document {i + 1} '{doc['url']}' of {len(links_docs)}")

                    doc_list = get_text_splits(doc["text"], "link")
                    doc_list = add_metadata(doc_list, "link", chatbot_id, doc["slug"], url=doc["url"])

                    #TODO: save url with source!


                    # embed html doc_list
                    embed_index(doc_list=doc_list, chatbot_id=chatbot_id)

        #TODO: js backend needs to be merged into this one
        # ping status endpoint
        
        express_api_endpoint = f"{app.config['api_url']}/api/chatbot/status/{chatbot_id}"

        #express_api_endpoint = f"{api_url}/api/chatbot/status/{chatbot_id}"



        try:
            response = requests.put(express_api_endpoint, json={'status': 'ready'})

            if response.status_code == 200:
                app.logger.info("Express API updated successfully!")
            else:
                app.logger.error(f"Failed to update Express API {express_api_endpoint}")

        except Exception as e:
            app.logger.error(f"Failed to update Express API {express_api_endpoint}")
            app.logger.error(e)


        return 'Files uploaded successfully'
    except Exception as e:
        app.logger.error(e)

        #TODO: log traceback!
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': 'Something went wrong!'}), 400


#------------------

class ReviseAnswerRequest(BaseModel):
    revisedText: str = Field(None, description="""The new revised text""")
    chatbotSlug: str = Field(None, description="""The chatbot id""")

@app.post('/bot/revise-answer', summary="", tags=[])
def revise2(body: ReviseAnswerRequest):
    """

    """
    revised_text = body.revisedText
    chatbot_id = body.chatbotSlug

    if not revised_text:
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter revisedText!'
        }), 400

    if not chatbot_id:
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter chatbotSlug!'
        }), 400

    doc_list = get_text_splits(revised_text)
    doc_list = add_metadata(doc_list, "revised_text", chatbot_id, "text")
    embed_index(doc_list=doc_list, chatbot_id=chatbot_id)
    return jsonify({
        'status': 'success',
        'message': 'Answer revised successfully!'
    })

#------------------

def clean_history(hist: List[Dict[str, str]]) -> str:
    out = ''
    for qa in hist[-5:]: # only the last 5
        if len(qa['bot']) < 2:
            continue
        out += 'user: ' + qa['user'] + '\nassistant: ' + qa['bot'] + "\n\n"
    return out



def get_prices(model_name) -> Dict[str, float]:
    """
    prices in Ct. per 1000 tokens
    """
    match model_name:

        # Azure OpenAI
        case 'gpt-35-turbo':
            inCt  = 0.15
            outCt = 0.2

        # OpenAI
        case 'gpt-3.5-turbo-16k': 
            inCt  = 0.3
            outCt = 0.4

        case 'gpt-3.5-turbo-0125':
            inCt  = 0.05
            outCt = 0.15

        case 'gpt-4':
            inCt  = 3.0
            outCt = 6.0

        case 'gpt-4-32k':
            inCt  = 6.0
            outCt = 12.0

        case 'gpt-4-0125-preview':
            inCt  = 1.0
            outCt = 3.0

        case _:
            inCt  = 1.0
            outCt = 1.0

    return {
        "inCt": inCt,
        "outCt": outCt
    }



def query_log(chatbot_id, queryId, sess, temperature, q, a, rating, llm, dura, sources, inputTokens, inCt, outputTokens, outCt):
    """
    Add a doc to nextsearch_log
    """

    connections.create_connection(hosts=app.config['elastic_uri'])

    # create the mappings in elasticsearch
    NextsearchLog.init()

    totalCt = ((inputTokens / 1000) * inCt) + ((outputTokens / 1000) * outCt)
    esdoc = {
        'queryid': queryId,
        'chatbotid': chatbot_id,
        'timest': datetime.now(),

        'date': date.today().isoformat(),

        'session': sess,
        'temperature': temperature,
        'q': q,
        'a': a,
        'rating': rating,
        'reason': '',
        'reasontags': '',
        'llm': llm,
        'durasecs': dura,
        'sources': sources,
        'inToks': inputTokens, 
        'inCt': inCt, 
        'outToks': outputTokens, 
        'outCt': outCt,
        'totalCt': totalCt
    }

    client = Elasticsearch(app.config['elastic_uri'])

    resp = client.index(index='nextsearch_log', document=esdoc)
    #TODO: check resp for success

    #print(resp)
    app.logger.info(resp)

    return resp




def get_llm(temperature: float, stream_key: str, sid: str):
    """
    Get the right LLM
    """
    if OPENAI_API_TYPE == 'azure':
        llm = AzureChatOpenAI(
            openai_api_version=OPENAI_API_VERSION, 
            deployment_name=OPENAI_DEPLOYMENT_NAME, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,

            openai_api_key=OPENAI_API_KEY,
            model_name=OPENAI_MODEL_NAME,
            temperature=temperature,
            streaming=True,
            callbacks=BaseCallbackManager([StreamingCallback(stream_key, sid)])
        )
    else:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=OPENAI_MODEL_NAME,
            temperature=temperature,
            streaming=True,
            callbacks=BaseCallbackManager([StreamingCallback(stream_key, sid)])
        )

    return llm


class QueryRequest(BaseModel):
    queryId: str = Field("", description="""The query id""") #generated by the js backend atm
    key: str = Field("", description="""String used for the streaming of the chat""")

    prompt: str = Field(None, description="""The prompt/question to the bot""")
    history: List[Dict[str,str]] = Field([], description="""""")
    chatbotSlug: str = Field(None, description="""The chatbot id. Example: 'MyBot_c2wun1'""")
    temprature: float = Field(0.1, description="""The temperature value passed to OpenAI affecting the strictness of it#s answers""")
    sid: str = Field(None, description="""String used for the streaming of the chat""")
    systemPrompt: str = Field("Antworte freundlich, mit einer ausführlichen Erklärung, sofern vorhanden auf Basis der folgenden Informationen. Please answer in the language of the question.", description="""A prompt always contextualizing the query used""")


@app.post('/bot/query', summary="Query the bot via prompt", tags=[])
def bot_query(body: QueryRequest):
    """
    The main route to use the chatbots LLM with a given prompt string, temperature, system prompt and history context
    """
    dura = datetime.now().timestamp()

    queryId = body.queryId
    prompt = body.prompt
    history = clean_history(body.history)
    chatbot_id = body.chatbotSlug
    system_prompt = body.systemPrompt
    temperature = body.temprature #typo in 'temprature' instead of is key temperature
    key = body.key
    sid = body.sid

    stream_key = key if key else f"{chatbot_id}_stream"

    sess = str(request.user_agent) + ' ' + str(request.environ.get('HTTP_X_REAL_IP', request.remote_addr)) +' '+ str(request.remote_addr)
    sessMD5 = hashlib.md5(sess.encode()).hexdigest()

    #TODO: we need a better way to create these ids... it seems kind of random
    if (queryId == None) or (queryId == ''):
        queryId = sessMD5


    encoding = tiktoken.encoding_for_model(OPENAI_MODEL_NAME)

    if not chatbot_id:
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter chatbotSlug!'
        }), 400
    if not prompt:
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter prompt!'
        }), 400
    if not sid:
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter sid in query!'
        }), 400


    default_temperature = 0.1
    temperature = temperature if temperature is not None else default_temperature
    
    llm = get_llm(temperature, stream_key, sid)

    prompt_template = system_prompt + """
    <ctx>    
        {context}
    </ctx>
    <hs>
    """ + history + """
    </hs>    
    Question: {question}
    """

    chat_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    index = determine_index(chatbot_id)

    db = ElasticsearchStore(
        es_url=app.config['elastic_uri'],
        index_name=index, 
        distance_strategy="COSINE",
        embedding=EMBEDDING
    )

    k = int(LLM_PAYLOAD / CHUNK_SIZE) - 1
    if (k < 2):
        k = 2

    scoredocs = db.similarity_search_with_score(prompt, k=k+10)

    query = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=False,
        return_source_documents=True,
        retriever=db.as_retriever(search_kwargs={'k': k}),
        chain_type_kwargs={"prompt": chat_prompt}
    )

    inputTokens = 0
    outputTokens = 0

    with get_openai_callback() as cb:
        qares = query.invoke({'query': prompt})
        qadocs = qares['source_documents'] #TODO: STS: deliver doc names and page numbers in the future
        inputDocTxt = ''

        sources = []
        count = 0
        for qadoc in qadocs:
            mdata = qadoc.metadata
            if 'chatbotId' in mdata:
                del mdata['chatbotId']

            nextScore = 0.0
            for scoredoc in scoredocs:
                if (len(qadoc.page_content) > 20) and (len(scoredoc[0].page_content) > 20) and (qadoc.page_content[:20] == scoredoc[0].page_content[:20]):
                    nextScore = scoredoc[1]
                    inputDocTxt += ' ' + qadoc.page_content
                    break

            # Lets make Percent of the score, only look at 0.6-1.0
            nextScore = float((nextScore - 0.6) * 250)
            if nextScore < 1.0:
                nextScore = 1.0
            if nextScore > 99.99:
                nextScore = 99.99

            mdata['score'] = round(nextScore, 2)
            sources.append(mdata)
            count += 1

        answer = qares['result']
        #print(f"Total Tokens: {cb.total_tokens}")
        #print(f"Prompt Tokens: {cb.prompt_tokens}")
        #print(f"Completion Tokens: {cb.completion_tokens}")

        app.logger.info("ANSWER: " + answer)

        #print(ans, flush=True)

        inputTokens = len(encoding.encode(inputDocTxt + ' ' + prompt_template))
        outputTokens = len(encoding.encode(answer))

        app.logger.info(f"Input Tokens: {inputTokens}")
        app.logger.info(f"Output Tokens: {outputTokens}")
        app.logger.info(f"Total Cost (USD): ${cb.total_cost}")



    d = get_prices(OPENAI_MODEL_NAME)
    inCt = d["inCt"]
    outCt = d["outCt"]

    # log question/answer
    dura = round(datetime.now().timestamp() - dura, 2)
    resp = query_log(chatbot_id,
              queryId,
              sessMD5,
              temperature,
              prompt,
              answer,
              0,
              OPENAI_MODEL_NAME,
              dura,
              sources,
              inputTokens,
              inCt,
              outputTokens,
              outCt)
    
    app.logger.info(resp)

    sources_index = "chatbot_" + chatbot_id

    client = Elasticsearch(app.config['elastic_uri'])

    s = Search(using=client, index=sources_index)
    s = s[0:10000]
    response = s.execute()
    srcs = (x.to_dict() for x in response.hits)
    src_grps = group_by([lambda d: d["metadata"]["sourceType"] ], srcs)

    #def print_type(x):
        
    new_sources = []
    for source in sources:
        app.logger.info("Source: " + repr(source))
        match source["sourceType"]:
            case "text":
                if "txt_id" in source:
                    source["text"] = ""
                    d2 = group_by([lambda d: d["metadata"]["txt_id"] ], src_grps["text"])
                    for src_item in d2[source["txt_id"]]:
                        source["text"] += " " + src_item["text"]

                new_sources.append(source)

            case "link":
                if "sourceFileId" in source:
                    source["text"] = ""
                    d2 = group_by([lambda d: d["metadata"]["sourceFileId"] ], src_grps["link"])
                    for src_item in d2[source["sourceFileId"]]:
                        source["text"] += " " + src_item["text"]
                        if "url" in src_item:
                            source["url"] = src_item["url"]

                new_sources.append(source)

            case "file":
                if "sourceFileId" in source:
                    source["text"] = ""
                    d2 = group_by([lambda d: d["metadata"]["sourceFileId"] ], src_grps["file"])
                    for src_item in d2[source["sourceFileId"]]:
                        source["text"] += " " + src_item["text"]
                        if "filename" in src_item:
                            source["filename"] = src_item["filename"]

                new_sources.append(source)


    if resp.body["result"] == "created":
        return jsonify({
            'status': 'success',
            'answer': answer,
            'query_id': queryId,
            'sources': new_sources #sources
        })
    else:
        return jsonify({
            'status': 'error',
            'message': resp.body["result"]
        }), 400


#------------------
    
#TODO create separate delete bot route

class DeleteBotRequest(BaseModel):
    chatbot_id: str = Field(None, description="""Chatbot id""")

@app.delete('/bot', summary="", tags=[])
def delete_bot(body: DeleteBotRequest):
    """
    Not implemented yet

    Delete a chatbot via it's id
    """
    chatbot_id = body.chatbot_id

    # Ensure chatbotId is provided
    if not chatbot_id:
        app.logger.error('Missing required parameter chatbotSlug!')
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter chatbotSlug!'
        }), 400

    client = Elasticsearch(app.config['elastic_uri'])
    id2name = get_slugs_for_names(client)
    index = determine_index(chatbot_id)


    if not chatbot_id in id2name:
        app.logger.error("Missing associated chatbot name of this id: '%s'!" % chatbot_id)
        return jsonify({
            'status': 'error',
            'message': 'Chatbot id not found!'
        }), 404
    else:
        chatbot_name = id_value=id2name[chatbot_id]


    #TODO: delete index chatbot_<slug_id>
    try:
        client.indices.delete(index=index)
        app.logger.info("Deleted index '%s' !" % index)
    except:
        app.logger.error("Could not delete index '%s' !" % index)


    #TODO: delete associated doc from index chatbot
    #try:
    delete_by_id(client, index="chatbot", id_field_name="slug", id_value=chatbot_id)
    #    app.logger.info("Deleted chatbot '%s' data from index '%s' !" % (chatbot_id, "chatbot"))
    #except:
    #    app.logger.error("Could not delete data for '%s' in index 'chatbot' !" % chatbot_id)


    #TODO: delete associated doc from index settings
    #try:
    delete_by_id(client, index="settings", id_field_name="displayName", id_value=chatbot_name)
    #    app.logger.info("Deleted chatbot '%s' data from index '%s' !" % (id2name[chatbot_id], "settings"))
    #except:
    #    app.logger.error("Could not delete data for '%s' in index 'settings' !" % id2name[chatbot_id])


    #TODO: delete associated doc from index nextsearch_log
    #try:
    delete_by_id(client, index="nextsearch_log", id_field_name="chatbotid", id_value=chatbot_id)
    #    app.logger.info("Deleted chatbot '%s' data from index '%s' !" % (chatbot_id, "nextsearch_log"))
    #except:
    #    app.logger.error("Could not delete data for '%s' in index 'nextsearch_log' !" % chatbot_id)

    return "", 202


#------------------

#TODO: overloaded route... split into two or three, one for each resource
#server/routes/api/chatbot.js

#FE calls js BE: /api/chatbot/resources/abotycrsh1
#which calls bot BE



class DeleteResourceRequest(BaseModel):
    sourceType: str = Field(None, description="""Source type: ...link, text, file ?""")
    sourceId: str = Field(None, description="""Source id?""")
    chatbotSlug: str = Field(None, description="""Chatbot id""")

@app.delete('/bot/resources', summary="delete a bot or resource via it's id", tags=[])
def delete_resource(body: DeleteResourceRequest):
    """
    * delete a bot via it's id
    * delete files used as training source
    
    or other resources... unclear atm
    """
    source_type = body.sourceType
    source_id = body.sourceId
    chatbot_id = body.chatbotSlug

    # Validate presence of sourceType
    if not source_type:
        msg = 'sourceType is required!'
        app.logger.error(msg)
        return jsonify({
            'status': 'error',
            'message': msg
        }), 400

    # Ensure chatbotId is provided
    if not chatbot_id:
        app.logger.error('Missing required parameter chatbotSlug!')
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameter chatbotSlug!'
        }), 400

    # Apply criteria based on sourceType
    filter_criteria = {
        "bool": {
            "must": [
                {"match": {"metadata.sourceType": source_type}},
                {"match": {"metadata.chatbotId": chatbot_id}},
            ]
        }
    }
    if source_type != 'text':
        if not source_id:
            app.logger.error('Missing required parameter sourceId!')
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameter sourceId!'
            }), 400
        new_match: Dict[str, Dict[str, Any]] = {
            "match": {
                "metadata.sourceFileId": source_id
            }
        }
        filter_criteria["bool"]["must"].append(new_match)

    try:
        # Assuming delete method returns a status or raises an exception on failure
        app.logger.info(filter_criteria)

        index = determine_index(chatbot_id)

        store = ElasticsearchStore(
            es_url=app.config['elastic_uri'],
            index_name=index, 
            embedding=EMBEDDING
        )

        store.client.delete_by_query(index=index, query=filter_criteria)   
        
        # isDeleted = index.delete(filter=filter_criteria)
    except Exception as e:
        #TODO: Handle specific exceptions if possible

        app.logger.error(str(e))

        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


    msg = 'Resource deleted successfully!'
    app.logger.info(msg)
    return jsonify({
        'status': 'success',
        'message': msg
    })


#------------------


# Splits the text into small chunks of 150 characters
def get_pdf_splits(pdf_file: str) -> List:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    doc_list = []
    for pg in pages:
        pg_splits = text_split.split_text(pg.page_content)
        doc_list.extend(pg_splits)
    return doc_list


def get_text_splits(text: str, source: str="text") -> List:
    chunk_size = 1536
    chunk_overlap = 200

    #if source == "link":
    #    chunk_size = 1536
    #    chunk_overlap = 200

    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    doc_list = text_split.split_text(text)
    return doc_list


ESDocument = namedtuple('Document', ['page_content', 'metadata'])

def add_metadata(doc_list: List[str], source_type: str, chatbot_id: str, source_file_id, tags=[], url=None, filename=None, txt_id=None) -> List[ESDocument]:
    """
    
    """
    for i, doc in enumerate(doc_list):
        # If doc is a string, convert it to the Document format
        if isinstance(doc, str):
            doc = ESDocument(page_content=doc, metadata={})
            doc_list[i] = doc

        # Update the metadata
        updated_metadata = doc.metadata.copy()
        updated_metadata["chatbotId"] = chatbot_id
        updated_metadata["tags"] = ' | '.join(tags)

        match source_type:
            case "text":
                updated_metadata["sourceType"] = "text"
                if txt_id is not None:
                    updated_metadata["txt_id"] = txt_id

            case "revised_text":
                updated_metadata["sourceType"] = "revised_text"

            case "pdf_file" | "word_file" | "text_file":
                updated_metadata["sourceType"] = "file"
                updated_metadata["sourceFileId"] = source_file_id
                if filename is not None:
                    updated_metadata["filename"] = filename
                
            case "link":
                updated_metadata["sourceType"] = "link"
                updated_metadata["sourceFileId"] = source_file_id
                if url is not None:
                    updated_metadata["url"] = url

        # Update the document in the doc_list with new metadata
        doc_list[i] = ESDocument(
                        page_content=doc.page_content,
                        metadata=updated_metadata
                      )

    return doc_list



@app.errorhandler(500)
def server_error(error):
    app.logger.exception('An exception occurred during a request: ' + str(error))
    return 'Internal Server Error', 500



#JS Backend routes to reimplement:
# http://localhost:8000/api/chatbot/add-resources

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)



