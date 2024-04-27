# docker build -t container_upgraded -f container_upgraded/Dockerfile  .; docker run --rm --name  container_upgraded container_upgraded;
FROM python:3.9.0

WORKDIR code_correction/repositories/simple_tool

# # RUN apt-get update && \
# # apt-get install -y pkg-config libopenblas-dev liblapack-dev build-essential gfortran python-dev libfreetype6-dev libjpeg-dev libhdf5-dev liblzo2-dev libbz2-dev 
# # && \
# # pip install cython==0.23.4 && \
# RUN pip install --no-deps numpy==1.12.1 && \
#     pip install --no-deps scipy==0.19.0 && \
#     pip install --no-deps scikit-learn==0.18.1
COPY . .

RUN pip install --no-cache-dir -r container_upgraded/requirements.txt

# CMD [ "python", "./main.py" ]