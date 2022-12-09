FROM python:3.8

RUN useradd -ms /bin/bash mycoolmc

RUN apt-get update
RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev

WORKDIR /home/mycoolmc
RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -
RUN echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install spot libspot-dev spot-doc python3-spot

WORKDIR /home/mycoolmc
RUN git clone https://github.com/moves-rwth/storm.git
WORKDIR /home/mycoolmc/storm
RUN mkdir build
RUN pwd
RUN ls
WORKDIR /home/mycoolmc/storm/build

RUN cmake ..
RUN make -j 1
WORKDIR /home/mycoolmc
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    maven \
    uuid-dev \
    python3 \
    virtualenv

RUN git clone https://github.com/moves-rwth/pycarl.git
WORKDIR /home/mycoolmc/pycarl
RUN python3 setup.py build_ext --jobs 1 develop

WORKDIR /home/mycoolmc

RUN git clone https://github.com/moves-rwth/stormpy.git
WORKDIR /home/mycoolmc/stormpy
RUN python3.8 setup.py build_ext --storm-dir /storm/build/ --jobs 1 develop

WORKDIR /home/mycoolmc
COPY requirements.txt .
RUN pip3.8 install -r requirements.txt


COPY common common
COPY custom_openai_gyms custom_openai_gyms
COPY openai_gym_training openai_gym_training
COPY safe_gym_training safe_gym_training
COPY unit_testing unit_testing
COPY verify_rl_agent verify_rl_agent
#COPY control_ui.py .
COPY taxi_abstraction.json .
COPY cool_mc.py .
COPY start_ui.sh .

RUN chmod -R 777 /home/mycoolmc

RUN apt-get install -y iproute2

ENTRYPOINT /bin/bash