FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

MAINTAINER Mingfei Sun

ARG UID
RUN useradd -u $UID --create-home mingfei
USER mingfei
WORKDIR /home/mingfei

# install all requirements
USER root
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

USER mingfei
ENV SC2PATH /home/mingfei/marl_ppo/3rdparty/StarCraftII

RUN echo 'set editing-mode vi' >> /home/mingfei/.inputrc
RUN echo 'set keymap vi' >> /home/mingfei/.inputrc

WORKDIR /home/mingfei/marl_ppo

