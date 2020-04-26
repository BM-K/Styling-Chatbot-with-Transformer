FROM ufoym/deepo:pytorch
# https://github.com/Beomi/deepo-nlp/blob/master/Dockerfile
# Install JVM for Konlpy
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    openjdk-8-jdk wget curl git python3-dev \
    language-pack-ko

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

# Install zsh
RUN apt-get install -y zsh && \
    sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"





# Install another packages
RUN pip install autopep8
RUN pip install  soynlp konlpy
RUN pip install torchtext pytorch_pretrained_bert
# Install dependency of styling chatbot
RUN pip install hgtk chatspace

# Add Mecab-Ko
RUN curl -L https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash
# install styling chatbot by BM-K
RUN git clone https://github.com/km19809/Styling-Chatbot-with-Transformer.git
RUN pip install -r Styling-Chatbot-with-Transformer/KoBERT/requirements.txt
RUN pip install ./Styling-Chatbot-with-Transformer/KoBERT

# Add non-root user
RUN adduser --disabled-password --gecos "" user

# Reset Workdir
WORKDIR /Styling-Chatbot-with-Transformer
