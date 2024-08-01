# Ollama bot

After deploy:

## WebUI for Ollama:
* http://localhost:8888
* use to install models like llama2, llama3 (https://ollama.com/library)

## Frontend
* simple FE: http://localhost:5000/

### Stack
* Gitea actions + Google lighthouse
* Gitea actions + Playright
* Nuxt.js + Bootstrap 5


## Backend:
* http://localhost:5000/openapi/swagger
* http://localhost/backend/openapi/swagger

### Stack
* FastAPI
* RabbitMQ/Kafka?
* OpenSearch


### Push image

```bash
docker login registry.tobiasweise.dev
docker-compose push

#sudo docker tag llm-python-backend nucberlin:5123/llm-python-backend
#sudo docker push nucberlin:5123/llm-python-backend
```

----

## Ideas

### Knowledge graph creation

https://www.linkedin.com/posts/sivas-subramaniyan_microsoft-research-is-bullish-on-the-concept-activity-7194953376470638592-dQ-U/?utm_source=share&utm_medium=member_desktop


clean dangling images

sudo docker rmi $(sudo docker images -f "dangling=true" -q)




