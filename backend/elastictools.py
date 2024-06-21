"""
Some helper functions to make querying easier
"""
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
import json
from elasticsearch import NotFoundError, Elasticsearch # for normal read/write without vectors
from elasticsearch_dsl import Search, A, UpdateByQuery, Document, Date, Integer, Keyword, Float, Long, Text, connections


def get_by_id(client: Elasticsearch, index: str, id_field_name: str, id_value: str):
    response = Search(using=client, index=index).filter("term", **{id_field_name: id_value})[0:10000].execute()
    return [hit.to_dict() for hit in response]


def update_by_id(client: Elasticsearch, index: str, id_field_name: str, id_value: str, values_to_set: Dict[str, Any]) -> None:
    #create painless insert script
    source = ""
    for k, v in values_to_set.items():
        source += f"ctx._source.{k} = {json.dumps(v)};"

    """
    body = {
        "query": { 
            "term": {
                id_field_name: id_value
            }
        },
        "script": {
            "source": source,
            "lang": "painless"
        }
    }
    client.update_by_query(index=index, body=body)
    """

    ubq = UpdateByQuery(using=client, index=index) \
        .query("term", **{id_field_name: id_value})   \
        .script(source=source, lang="painless")

    response = ubq.execute()
    return response.success()



def delete_by_id(client: Elasticsearch, index: str, id_field_name: str, id_value: str):
    s = Search(using=client, index=index).filter("term", **{id_field_name: id_value})
    response = s.delete()
    #if not response.success():
    #    raise Exception("Unable to delete id '%s' in index '%' !" % (index, id_value))

    print(response, flush=True)



def get_datetime_interval(search: Search, start, end) -> Search:
    return search.filter("range", timest={"gte": start}).filter("range", timest={"lte": end})


