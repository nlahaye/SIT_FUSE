
FROM nvcr.io/nvidia/pytorch:24.03-py3
 
WORKDIR /app

RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime; DEBIAN_FRONTEND=noninteractive; apt-get update
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install software-properties-common
RUN DEBIAN_FRONTEND=noninteractive; add-apt-repository ppa:ubuntugis/ppa
RUN DEBIAN_FRONTEND=noninteractive; apt-get update
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install gdal-bin
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install libgdal-dev
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL
RUN pip install GPUtil geopandas geocube dask_ml zarr scikit-image pyresample
RUN pip install netCDF4 ray tabulate "ray[tune]"

RUN chmod -R a+rwx /app


