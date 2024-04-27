# docker build -t simple_tool:container_original -f repositories/simple_tool/container_original/Dockerfile  .; docker run --rm --name  simple_tool:container_original simple_tool:container_original;

# This is a docker file that serves a version of python from 6 years ago that has numpy, scipy, and sklearn installed already
# We can look at the contents of this file by using `docker history --no-trunc  morpheo/sklearn-base:latest >> /tmp/history`
FROM morpheo/sklearn-base:latest

WORKDIR code_correction/repositories/simple_tool
COPY . .
