#!/bin/bash
# A script for running the docker container and experiements. Run from inside
# Domain-Guided-Monitoring

mkdir -p ~/Domain-Guided-Monitoring/mlruns

mkdir -p ~/Domain-Guided-Monitoring/artifacts

docker build -t domainml ~/Domain-Guided-Monitoring/.

docker run -v ~/Domain-Guided-Monitoring/mlruns:/Domain-Guided-Monitoring/mlruns \
-v ~/Domain-Guided-Monitoring/artifacts:/Domain-Guided-Monitoring/artifacts  \
-v ~/Domain-Guided-Monitoring/data:/Domain-Guided-Monitoring/data \
-v ~/Domain-Guided-Monitoring/src:/Domain-Guided-Monitoring/src \
-v ~/Domain-Guided-Monitoring/Makefile:/Domain-Guided-Monitoring/Makefile \
-v ~/Domain-Guided-Monitoring/main.py:/Domain-Guided-Monitoring/main.py \
--rm  \
--name domainml \
-td \
domainml

# You can run the Mlflow server from this docker container. I recommend
# having a separate experiment container because your server will be really slow
# if you run it in the experiment container
docker run -v ~/Domain-Guided-Monitoring/mlruns:/Domain-Guided-Monitoring/mlruns \
-v ~/Domain-Guided-Monitoring/artifacts:/Domain-Guided-Monitoring/artifacts  \
-v ~/Domain-Guided-Monitoring/data/knowledge:/Domain-Guided-Monitoring/data/knowledge \
-v ~/Domain-Guided-Monitoring/src:/Domain-Guided-Monitoring/src \
-v ~/Domain-Guided-Monitoring/Makefile:/Domain-Guided-Monitoring/Makefile \
-v ~/Domain-Guided-Monitoring/main.py:/Domain-Guided-Monitoring/main.py \
--rm  \
--name domainml-ui \
-p 5000:5000 \
-td \
domainml