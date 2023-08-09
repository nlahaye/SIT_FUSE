#!/bin/bash

image=sit_fuse_interactive

apptainer build ${image}.sif ${image}.def

if [ $? -ne 0 ] ; then
    echo "Trouble with apptainer build"
    exit 1
fi


