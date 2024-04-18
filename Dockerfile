# Base image
FROM public.ecr.aws/docker/library/ubuntu:22.04

# Set ENV
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/opt/aws/neuron/bin:$PATH

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    gnupg2 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/tmp* \
    && apt-get clean

# Set driver
RUN echo "deb https://apt.repos.neuron.amazonaws.com focal main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

RUN apt-get update \
    && apt-get install -y \
    aws-neuronx-tools \
    aws-neuronx-runtime-lib \
    aws-neuronx-collectives \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/tmp* \
    && apt-get clean

# Set pip
RUN pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Set working directory
WORKDIR /app

# Copy requirements file
COPY ./app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY ./app .

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
