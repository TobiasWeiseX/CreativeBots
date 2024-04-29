import os
from elasticsearch_dsl import Document, InnerDoc, Date, Integer, Keyword, Float, Long, Text, connections, Object

# Define a default Elasticsearch client
connections.create_connection(hosts="http://localhost:9200")

class Article(Document):
    title = Text(analyzer='snowball', fields={'raw': Keyword()})
    body = Text(analyzer='snowball')
    tags = Keyword()
    published_from = Date()
    lines = Integer()

    class Index:
        name = 'blog'
        settings = {
          "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        self.lines = len(self.body.split())
        return super(Article, self).save(** kwargs)


#======= nextsearch_log ===========

class Sources(InnerDoc):
    score = Float()
    sourceFileId = Text()
    sourceType = Text()
    tags = Text()

class NextsearchLog(Document):
    a = Text()
    chatbotid = Keyword()
    durasecs = Float()
    inCt = Float()
    inToks = Long()
    llm = Text()
    outCt = Float()
    outToks = Long()
    q = Text()
    queryid = Keyword()
    rating = Long()
    reason = Text()
    reasontags = Text()
    session = Keyword()

    sources = Object(Sources) #Text(analyzer='snowball')
    temperature = Float()
    totalCt = Float()

    timest = Date() #timestamp
    date = Date() #iso date

    class Index:
        #name = 'test_nextsearch_log'
        name = 'nextsearch_log'
        settings = {
            "number_of_shards": 1,
        }

    def save(self, ** kwargs):
        self.lines = len(self.body.split())
        return super(NextsearchLog, self).save(** kwargs)


if __name__ == "__main__":
    elastic_uri = os.getenv("ELASTIC_URI")
    #elastic_uri = "http://localhost:9200"
    assert elastic_uri

    # Define a default Elasticsearch client
    connections.create_connection(hosts=elastic_uri)
    #connections.create_connection(hosts)

    # create the mappings in elasticsearch
    NextsearchLog.init()

    # create the mappings in elasticsearch
    #Article.init()

    # create and save and article
    #article = Article(meta={'id': 42}, title='Hello world!', tags=['test'])
    #article.body = ''' looong text '''
    ##article.published_from = datetime.now()
    #article.save()

    #article = Article.get(id=42)
    #print(article.is_published())

    # Display cluster health
    #print(connections.get_connection().cluster.health())
