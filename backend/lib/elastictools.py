"""
Some helper functions to make querying easier
"""
import time, json, os
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
from elasticsearch import NotFoundError, Elasticsearch # for normal read/write without vectors
from elasticsearch_dsl import Search, A, UpdateByQuery, Document, Date, Integer, Keyword, Float, Long, Text, connections
from elasticsearch.exceptions import ConnectionError


def get_by_id(index: str, id_field_name: str, id_value: str):
    client = connections.get_connection()
    response = Search(using=client, index=index).filter("term", **{id_field_name: id_value})[0:10000].execute()
    return [hit.to_dict() for hit in response]


def update_by_id(index: str, id_field_name: str, id_value: str, values_to_set: Dict[str, Any]) -> None:
    client = connections.get_connection()
    #create painless insert script
    source = ""
    for k, v in values_to_set.items():
        source += f"ctx._source.{k} = {json.dumps(v)};"

    ubq = UpdateByQuery(using=client, index=index) \
        .query("term", **{id_field_name: id_value})   \
        .script(source=source, lang="painless")

    response = ubq.execute()
    return response.success()


def delete_by_id(index: str, id_field_name: str, id_value: str):
    client = connections.get_connection()
    s = Search(using=client, index=index).filter("term", **{id_field_name: id_value})
    response = s.delete()
    #if not response.success():
    #    raise Exception("Unable to delete id '%s' in index '%' !" % (index, id_value))
    print(response, flush=True)



def get_datetime_interval(search: Search, start, end) -> Search:
    return search.filter("range", timest={"gte": start}).filter("range", timest={"lte": end})


#schema intro spection and maybe comparison/diffing
#TODO: route that takes a schema json and compares to internal structure and returns boolean

def simplify_properties(d):
    new_d = {}
    for field, d3 in d["properties"].items():
        if "type" in d3:
            new_d[field] = d3["type"]
        elif "properties" in d3:
            new_d[field] = simplify_properties(d3)
    return new_d


def get_type_schema():
    client = connections.get_connection()
    d = client.indices.get(index="*").body
    new_d = {}
    for index, d2 in d.items():
        new_d[index] = simplify_properties(d2["mappings"])
    return new_d



def wait_for_elasticsearch():
    #TODO: find a clean way to wait without exceptions!
    #Wait for elasticsearch to start up!
    #elastic_url = os.getenv("ELASTIC_URI")
    #assert elastic_url
    i = 1
    while True:
        try:
            #client = Elasticsearch(hosts=elastic_url)
            client = connections.get_connection()
            client.indices.get_alias(index="*")
            #connections.create_connection(hosts=app.config['elastic_uri'])
            #connections.get_connection().cluster.health(wait_for_status='yellow')
            #init_indicies()
            print("Elasticsearch found! Run Flask-app!", flush=True)
            return
        except ConnectionError:
            i *= 2 #1.5
            time.sleep(i)
            print("Elasticsearch not found! Wait %s seconds!" % i, flush=True)





