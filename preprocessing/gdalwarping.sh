#!/bin/bash

targetfile=$1
infile=$2
outfile=$3

echo "Warping ${infile} using the SRS of ${targetfile} to ${outfile}"

projectionfile=$(mktemp /tmp/tif2targetsrs.XXXXXX)

gdalsrsinfo -o wkt "${targetfile}" > "${projectionfile}"
gdalwarp -t_srs "${projectionfile}" "${infile}" "${outfile}"
#rm "${projectionfile}"




