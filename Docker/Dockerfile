FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y python3 python3-pip
# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install numpy pandas matplotlib beautifulsoup4 openai PyMuPDF python-dotenv jinja2 google-ai-generativelanguage google-generativeai pyyaml

# Set the default command to Python3
CMD ["python3"]


# To build image from docker file
# docker build -t auto_survey --platform linux/amd64 .

# To run this image
# docker run -it auto_survey

# get container id from running image
# then exec with the following command
# docker exec -it container_id /bin/bash

# copy bundle to docker container
# docker cp /your/local/directory your_container_name:/app


# tag your image 
# docker tag auto_survey ihsaanullah/auto_survey:latest

# push image
# docker push ihsaanullah/auto_survey:latest

