FROM ubuntu:20.04


ARG DEBIAN_FRONTEND=noninteractive

# # Install linux dependencies
RUN apt-get update \
 && apt-get install -y software-properties-common git zip curl wget git gcc build-essential

WORKDIR /app/


# Install Psi Solver
RUN  apt -qq update &&  apt -qq install -y unzip xz-utils libxml2-dev 

# Install dlang
RUN curl -fsS https://dlang.org/install.sh | bash -s dmd
RUN echo "source ~/dlang/dmd-2.109.1/activate" >> /root/.bashrc

RUN git clone https://github.com/eth-sri/psi.git \
&& cd ./psi \
 && ./dependencies-release.sh \
 && ./build-release.sh \
 && mkdir bin \
 && mv psi ./bin

ENV PATH="/app/psi/bin:$PATH"


### Install GuBpi
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
 &&  dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb

RUN  apt-get update \
 &&  apt-get install -y apt-transport-https \
 &&  apt-get update \
 &&  apt-get install -y dotnet-sdk-7.0

RUN git clone https://github.com/gubpi-tool/gubpi.git \
 && cd /app/gubpi/vinci/ \
 && make -j 4 \
 && cd /app/gubpi/src/ \
 && dotnet restore . \
 && dotnet build -c release -o ../app --no-restore \
 && cp /app/gubpi/vinci/vinci /app/gubpi/app 

ENV PATH="/app/gubpi/app:$PATH"


RUN apt-get install -y python3 python3-pip

RUN pip3 install wheel
RUN git clone https://github.com/unitn-sml/wmi-pa.git \
 && cd wmi-pa \
 && pip3 install .

#  # Install Latte
RUN  apt-get update \
 &&  apt-get install -y  m4 cmake g++ make libgmp3-dev

# Install mathsat
RUN pip3 install pysmt && \
 pysmt-install --msat --confirm-agreement
RUN wmipa-install --nra

RUN wget -c "https://github.com/latte-int/latte/releases/download/version_1_7_5/latte-integrale-1.7.5.tar.gz" \
 && tar xvf latte-integrale-1.7.5.tar.gz \
 && cd latte-integrale-1.7.5\
 && ./configure --prefix=/app/latte --with-default=/app/latte \
 && make -j 8 \
 && make install
ENV PATH="$PATH:/app/latte/bin"

# Install Volesti
RUN  apt-get install -y lp-solve libboost-all-dev \
&& wmipa-install --volesti -yf
ENV PATH="/root/.wmipa/approximate-integration/bin:$PATH"

COPY ./ ./

# Install WMI-LP
RUN cd wmi-lp \
 && pip3 install .
