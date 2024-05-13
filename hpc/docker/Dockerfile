
FROM nvcr.io/nvidia/pytorch:24.03-py3
 
WORKDIR /app

RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime; DEBIAN_FRONTEND=noninteractive; apt-get update
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install software-properties-common
RUN DEBIAN_FRONTEND=noninteractive; add-apt-repository ppa:ubuntugis/ppa
RUN DEBIAN_FRONTEND=noninteractive; apt-get update
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install gdal-bin
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install libgdal-dev
RUN DEBIAN_FRONTEND=noninteractive; apt-get --assume-yes install git
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL GPUtil geopandas geocube dask_ml zarr scikit-image pyresample wandb
RUN pip install netCDF4 ray tabulate "ray[tune]" pytorch-lightning pyhdf h5py regionmask

RUN git clone https://github.com/nlahaye/x-transformers.git
RUN git clone https://github.com/nlahaye/I-JEPA.git
RUN git clone https://github.com/nlahaye/SIT_FUSE.git 
RUN git clone https://github.com/nlahaye/learnergy.git

RUN echo 'export PYTHONPATH=${PYTHONPATH}:/app/learnergy/:/app/SIT_FUSE/:/app/x-transformers/:/app/I-JEPA/:/app/byol-pytorch/' >> ~/.bashrc

RUN chmod -R a+rwx /app


