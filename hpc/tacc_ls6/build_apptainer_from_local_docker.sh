#!/bin/bash

image=sit_fuse

apptainer build ${image}.sif ${image}_from_local_docker.def

if [ $? -ne 0 ] ; then
    echo "Trouble with apptainer build"
    exit 1
fi



