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





#======= Query Log ===========


class Sources(InnerDoc):
    score = Float()
    #sourceFileId = Text()
    sourceType = Text()
    tags = Text()

    #new fields
    sourceFileId = Keyword()
    filename = Keyword()
    url = Keyword()
    txt_id = Keyword()
    page = Integer()



class QueryLog(Document):
    answer = Text()
    question = Text()

    chatbotid = Keyword()
    durasecs = Float()
    #inCt = Float()
    inToks = Long()
    llm = Text()
    #outCt = Float()
    outToks = Long()

    #queryid = Keyword()
    #rating = Long()
    #reason = Text()
    #reasontags = Text()
    session = Keyword()

    sources = Object(Sources)
    temperature = Float()
    #totalCt = Float()

    timest = Date() #timestamp
    date = Date() #iso date

    class Index:
        name = 'query_log'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        return super(QueryLog, self).save(**kwargs)



if __name__ == "__main__":
    elastic_uri = os.getenv("ELASTIC_URI")
    #elastic_uri = "http://localhost:9200"

    # create and save and article
    #article = Article(meta={'id': 42}, title='Hello world!', tags=['test'])
    #article.body = ''' looong text '''
    ##article.published_from = datetime.now()
    #article.save()

    #article = Article.get(id=42)
    #print(article.is_published())

    # Display cluster health
    #print(connections.get_connection().cluster.health())
