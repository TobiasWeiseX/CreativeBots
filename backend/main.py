from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from neo4j import GraphDatabase

import os, sys
from multiprocessing import Pool
from bs4 import BeautifulSoup
import requests


from webbot import * #Bot, innerHTML
from xing import *


env = Environment(loader=FileSystemLoader('templates'))
app = FastAPI()

class JobSearch(BaseModel):
    location: str
    language: str



def xing_job_search(location: str, radius: int) -> list:
    with Bot() as bot:
        vars_ = {
            "page": 1,
            "filter.industry%5B%5D": 90000,
            "filter.type%5B%5D": "FULL_TIME",
            "filter.level%5B%5D": 2,
            "location": location,
            "radius": radius
        }
        start_url = "https://www.xing.com/jobs/search?" + "&".join([k + "=" + str(v) for k, v in vars_.items()])


        def kill_cookie_questions():
            bot.click_id("consent-accept-button")


        def next_page():
            nav = bot.get_elements_by_tag_name("nav")[1]
            next_site_link = get_elements_by_tag_name(nav, "a")[-1]
            bot.click(next_site_link)

        def get_nr_pages():
            nav = bot.get_elements_by_tag_name("nav")[1]
            return int(get_elements_by_tag_name(nav, "a")[-2].text)

        def get_items():
            rs = []
            for article in bot.get_elements_by_tag_name("article"):
                rs.append( get_children(article)[0].get_attribute("href") )
            return rs

        return collect_pagination_items(bot, start_url, next_page, get_nr_pages, get_items, kill_cookie_questions)





"""
pwd = "neo4j2"
proto = "bolt"
host = "192.168.99.101"

driver = GraphDatabase.driver("%s://%s:7687" % (proto, host), auth=("neo4j", pwd), encrypted=False)

def add_friend(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
           name=name, friend_name=friend_name)

def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                         "RETURN friend.name ORDER BY friend.name", name=name):
        print(record["friend.name"])

with driver.session() as session:
    session.write_transaction(add_friend, "Arthur", "Guinevere")
    session.write_transaction(add_friend, "Arthur", "Lancelot")
    session.write_transaction(add_friend, "Arthur", "Merlin")
    session.read_transaction(print_friends, "Arthur")

driver.close()
"""


@app.post("/search")
def job_search(js: JobSearch):


    #https://berlinstartupjobs.com/?s=python&page=3

    location = "Berlin"
    radius = 50


    with Bot() as bot:
        vars_ = {
            "page": 1,
            "filter.industry%5B%5D": 90000,
            "filter.type%5B%5D": "FULL_TIME",
            "filter.level%5B%5D": 2,
            "location": location,
            "radius": radius
        }
        start_url = "https://www.xing.com/jobs/search?" + "&".join([k + "=" + str(v) for k, v in vars_.items()])

        bot.set_url(start_url)
        return bot.get_page_content()




@app.get("/")
async def root():
    template = env.get_template('index.twig')
    html = template.render()
    return HTMLResponse(html)



