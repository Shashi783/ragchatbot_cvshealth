### .env file
enter required secrets in .env as shown in env.example.

## Ingestion API
Check out the [Ingestion API README](core/ingest/README.md) for more information.

## Using Docker Compose

1. go to deployment folder
```sh
cd deployment
```

2. Running the Vector DB using Docker Compose
```sh
sudo docker compose up -d
```

3. Check whether Docker is healthy or not using this command:
```sh
sudo docker ps
```

4. Stopping Docker
```sh
sudo docker compose down
```


## Install Docker (If Not Already Installed)

1. Linux Installation:
```sh
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo systemctl status docker
```

#check if docker id is running
'''ps aux | grep dockerd'''
#If dockerd is not running, start it manually:
'''sudo dockerd'''










