
#!/bin/bash
if [ "$1" == "" ] ; then
  echo "usage: sh build.sh <version>"
  echo "Example:"
  echo "./build.sh latest"
  exit 0
fi

version=$1
image=sit_fuse

nvidia-docker build --build-arg \
  --network=host \
  -t ${image}:${version} .

if [ $? -ne 0 ] ; then
    echo "Trouble with nvidia-docker build"
    exit 1
fi



