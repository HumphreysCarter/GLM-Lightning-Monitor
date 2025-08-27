# GLM Lightning Monitor

![Monitor webpage example](docs/monitor_example.png)

## Installation

### Requirements
 * [Docker](https://docs.docker.com/engine/install/)

### Build the Container 

### Initial Install
```shell
$ git clone https://github.com/HumphreysCarter/GLM-Lightning-Monitor.git
$ cd GLM-Lightning-Monitor
$ docker compose up --build
```

### Updates
```shell
$ cd GLM-Lightning-Monitor
$ git pull
$ docker compose up --build
```

## Usage

Once running, the application will be available at [localhost:8080](http://localhost:8080/). API documentation is available at [localhost:8080/api/docs](http://localhost:8080/api/docs).