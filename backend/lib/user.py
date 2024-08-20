"""
All around managing users
"""
import os, json, hashlib, traceback, logging
from elasticsearch import NotFoundError, Elasticsearch # for normal read/write without vectors

from lib.models import User
from lib.elastictools import get_by_id, update_by_id, delete_by_id, wait_for_elasticsearch


elastic_url = os.getenv("ELASTIC_URI")

assert elastic_url


def hash_password(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def create_user(email, password, role="user", verified=False):
    user = User(meta={'id': email}, email=email, password_hash=hash_password(password + email), role=role)
    user.creation_date = datetime.now()
    user.isEmailVerified = verified
    user.save()
    return user


def create_default_users():
    #create default users
    client = Elasticsearch(elastic_url)
    default_users = os.getenv("DEFAULT_USERS")
    if default_users:
        for (email, pwd, role) in json.loads(default_users):
            if len(get_by_id(client, index="user", id_field_name="email", id_value=email)) == 0:
                create_user(email, pwd, role=role, verified=True)







