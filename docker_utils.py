import docker
import os
import time

CONTAINER_NAME = "core_model_container"
client = docker.from_env()


def delete_container(container_name):
    client = docker.from_env()
    container = client.containers.get(container_name)
    try:
        container.stop()
        container.remove()
    except docker.errors.APIError as e:
        # Handle 409 Conflict error
        if "already in progress" in str(e):
            print("Removal of container is already in progress. Retrying...")
            # Wait for a short time before retrying
            time.sleep(1)
            delete_container(container_name)
        else:
            raise


def is_container_running(container_name):
    client = docker.from_env()
    containers = client.containers.list()
    for container in containers:
        if container.name == container_name:
            return True
    return False


def container_exists(container_name):
    try:
        client.containers.get(container_name)
        return True
    except docker.errors.NotFound:
        return False


def start_docker():
    if is_container_running(container_name=CONTAINER_NAME):
        return
    container = client.containers.run(
        "tensorflow/serving",
        detach=True,
        ports={"8501/tcp": 8501},
        name=CONTAINER_NAME,
        volumes={
            os.getcwd()
            + "\\models\\core_model": {
                "bind": "/models/core_model",
                "mode": "rw",
            }
        },
        environment={"MODEL_NAME": "core_model"},
        tty=True,
    )


def cleanup():
    if is_container_running(CONTAINER_NAME):
        delete_container(CONTAINER_NAME)
