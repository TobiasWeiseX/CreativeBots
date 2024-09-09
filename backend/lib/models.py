import os
from elasticsearch_dsl import Document, InnerDoc, Nested, Date, Integer, Keyword, Float, Long, Text, connections, Object, Boolean

class User(Document):
    creation_date = Date()
    email = Keyword()
    password_hash = Text(index=False)
    role = Keyword()

    #salt = Text(index=False)
    #profileImage = Text(index=False)
    #profileImage = Keyword()

    isEmailVerified = Boolean()
    #status = Text()

    #otpExpires = Date()
    #resetPasswordToken = Text(index=False)
    #mailToken = Text(index=False)

    class Index:
        name = 'user'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(User, self).save(**kwargs)


class Chatbot(Document):
    creation_date = Date()
    changed_date = Date()
    name = Text()
    creator_id = Keyword()
    description = Text()
    systemPrompt = Text(index=False)

    #slug = Keyword()
    files = Nested()
    text = Text()
    links = Nested()

    #chatbotImage = Text(index=False)
    sourceCharacters = Integer()

    visibility = Keyword() #public, private, group?
    #status = Keyword()

    temperature = Float()
    llm_model = Keyword()


    class Index:
        name = 'chatbot'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(Chatbot, self).save(**kwargs)


class Text(Document):
    creation_date = Date()
    creator_id = Keyword()
    text = Text()
    md5 = Keyword()

    class Index:
        name = 'text'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(Text, self).save(**kwargs)


class Question(Document):
    question = Text(index=False, required=True)
    md5 = Keyword()

    class Index:
        name = 'question'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(Question, self).save(**kwargs)


class Answer(Document):
    question_id = Keyword()
    answer = Text(index=False, required=True)
    md5 = Keyword()

    class Index:
        name = 'answer'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(Answer, self).save(**kwargs)


class LogEntry(Document):
    message = Text(index=False, required=True)
    level = Keyword() #Integer(required=True)
    creation_time = Date()


    name = Keyword()

    # 'args': ('GET /socket.io/?EIO=4&transport=websocket&sid=MtyTmZQs5IA6DnvhAAAA HTTP/1.1', '200', '-'), 

    pathname = Keyword()
    #  'pathname': '/usr/local/lib/python3.12/dist-packages/werkzeug/_internal.py', 

    filename = Keyword()
    # 'filename': '_internal.py', 

    module = Keyword()
    # 'module': '_internal',

    lineno = Integer(required=True)
    # 'lineno': 97, 

    funcName = Keyword()
    # 'funcName': '_log',


    #  'created': 1725709403.1972203,
    #  'msecs': 197.0, 

    threadName = Keyword()
    # 'threadName': 'Thread-15 (process_request_thread)',

    processName = Keyword()
    #  'processName': 'MainProcess',





    class Index:
        name = 'logentry'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(LogEntry, self).save(**kwargs)


#======= Query Log ===========


#class Sources(InnerDoc):
#score = Float()
#tags = Text()
#filename = Keyword()
#page = Integer()


#----------------------------------------------

def init_indicies():
    """
    Create the mappings in elasticsearch
    """
    for Index in [LogEntry, Question, Answer, Chatbot, User, Text]:
        Index.init()


