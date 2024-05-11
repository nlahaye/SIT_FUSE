---
description: This page describes how to use the Dockerized environment
---

# Dockerized Environment

Make sure nvidia-docker is installed on the machine you wish to run on

```
cd <path_to_SIT_FUSE>/SIT_FUSE/hpc/docker
```

```
./build.sh latest
```

See Docker's documentation for futher details on running containers:\
[https://docs.docker.com/guides/walkthroughs/run-a-container/](https://docs.docker.com/guides/walkthroughs/run-a-container/) \


You can also run with regular docker, instead of nvidia-docker, but it will not be optimized for GPU utilization:

```
docker build -t sit_fuse:latest .
#In same directory as above example
```

The image has SIT\_FUSE installed in the /app director
