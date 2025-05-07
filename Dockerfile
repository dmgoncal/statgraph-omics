FROM r-base:4.4.3

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
ENV PYTHON_VERSION=3.10.14

# Install system packages for dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    build-essential \
    libpq-dev \
    git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN apt update && apt install -y \
    build-essential libreadline-dev libx11-dev libxt-dev \
    libcairo2-dev libssl-dev libbz2-dev libzstd-dev liblzma-dev \
    libcurl4-openssl-dev libxml2-dev libfontconfig1-dev \
    libharfbuzz-dev libfribidi-dev libfreetype6-dev \
    libpng-dev libtiff5-dev libjpeg-dev \
    texinfo texlive texlive-fonts-extra zlib1g-dev \
    libgit2-dev

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

# Initialize conda for the bash shell
RUN /opt/conda/bin/conda init bash

# Create conda environment with Python 3.10.14
RUN conda create -y -n pyenv python=${PYTHON_VERSION} && \
    conda clean -afy

# Activate conda environment by default for all subsequent RUN commands
SHELL ["conda", "run", "-n", "pyenv", "/bin/bash", "-c"]

# Upgrade pip within the conda environment
RUN pip install --upgrade pip

# Set PYTHONPATH (you can adjust if needed)
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set the working directory
WORKDIR /app

# Copy your local cloned repo into the container
COPY . /app

# Install Python libraries using pip within conda environment
RUN pip install -r requirements.txt

# Install R libraries using Rscript
RUN Rscript requirements.r

# Default command to start bash shell
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate pyenv && exec bash"]

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate pyenv" >> ~/.bashrc

