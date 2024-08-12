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
```










