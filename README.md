# Creative Bots

* Disclaimer: Very much alpha software, biased to my own deployment needs
* License: GPL3

* Home repo: https://git.tobiasweise.dev/Tobias/CreativeBots
* Mirror: https://github.com/TobiasWeiseX/CreativeBots


## Configuration
* create a .env file in the deployment directory based on sample.env


## Build & Deploy via Docker-compose

```bash
cd deployment
docker-compose build
docker-compose up -d
```

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





