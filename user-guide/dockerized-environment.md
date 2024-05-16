---
description: This page describes how to use the Dockerized environment
---

# Dockerized Environment

## NVIDIA-Docker Build

```
cd <path_to_SIT_FUSE>/SIT_FUSE/hpc/docker
```

```
./build.sh latest
```

See Docker's documentation for futher details on running containers:\
[https://docs.docker.com/guides/walkthroughs/run-a-container/](https://docs.docker.com/guides/walkthroughs/run-a-container/) \\

### NVIDIA Container Toolkit + Docker

The nvidia-docker wrapper has been deprecated in favor of the NVIDIA Container Tookit. You can read more about installation and setup here:\
[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Once initial setup is complete, build the image with this command

```
docker build -t sit_fuse:latest .
#In same directory as above example
```

To run the container, you will need a few special arguments (namely  --runtime=nvidia -e NVIDIA\_VISIBLE\_DEVICES=nvidia.com/gpu=all). The proper values to use, for your specific case are detailed in the NVIDIA Container Toolkit link above.

{% code fullWidth="true" %}
```
docker run -it -v /data/:/data/ \
--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all sit_fuse
```
{% endcode %}

The image has SIT\_FUSE installed in the /app director
