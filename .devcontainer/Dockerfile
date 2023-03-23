FROM python:3

WORKDIR /usr/src/app

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip3 install --pre --upgrade astroquery[all]