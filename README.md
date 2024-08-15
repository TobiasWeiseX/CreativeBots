# Creative Bots


## Configuration
* create a .env file in the deployment directory based on sample.env


## Build & Deploy via Docker-compose

```bash
cd deployment
docker-compose build
docker-compose up -d
```

After deploy:

### WebUI for Ollama:
* http://localhost:8888
* use to install model llama3 (or more https://ollama.com/library)

----

## Usage

### Frontend
* simple FE: http://localhost:5000/

### Backend:
* http://localhost:5000/openapi/swagger
* http://localhost/backend/openapi/swagger


----

## Stack ideas

### Frontend
* Gitea actions + Google lighthouse
* Gitea actions + Playright
* Nuxt.js + Bootstrap 5

### Backend
* FastAPI
* RabbitMQ/Kafka?
* OpenSearch





