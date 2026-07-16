FROM continuumio/miniconda3:latest

COPY environment.yml /environment.yml
RUN conda env create -f /environment.yml
ENV PATH="/opt/conda/envs/bayes_hfs-dev/bin:$PATH"
ENV CONDA_DEFAULT_ENV="bayes_hfs-dev"
RUN echo "conda activate bayes_hfs-dev" >> ~/.bashrc
RUN pip install bayes_hfs