# datasci-model-api

This repository is the deployment of a model that predict PM2.5 3 days in advance

# Running

This application has implemented with fastapi

### Building Image

`docker build . -t datasci-api`

### Start Server port 8080

`docker-compose up`

### Deployment

`gcloud config set project datasci-line-api`

`gcloud builds submit --tag gcr.io/datasci-line-api/datasci-model-api`

`gcloud beta run deploy --image gcr.io/datasci-line-api/datasci-model-api --platform managed`
