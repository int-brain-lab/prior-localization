FROM bdgercek/iblenv
LABEL maintainer="Berk Gercek"
LABEL version="0.1"
LABEL description="IBL environment container with all prior-localization scripts"

WORKDIR /code
COPY . ./prior-localization/
COPY ./.one_params /root/
RUN mamba install dask=2021.4.0
RUN mamba install -c intel mpi4py
RUN mamba install -c conda-forge dask-mpi dask-jobqueue
VOLUME ["/data/flatiron/", "/data/analysis/"]