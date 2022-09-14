# HANDWRITING RECOGNITION

This repository contains the code for the Final Project of the Internet of Things course in the Artificial Intelligence master degree at UniBo.
## Project description

Machine learning and deep learning have become indispensable part of the existing technological
domain. Edge computing and Internet of Things (IoT) together presents a new opportunity to
imply machine learning techniques and deep learning models at the resource constrained by
embedded devices at the edge of the network. Generally, machine learning and deep learning
models requires enormous amount of power to predict a scenario. The aim of TinyML paradigm is
to shift from traditional high-end systems to low-end clients. The challenges doing such transition
are in particularly to maintain the accuracy of learning models, provide train-to-deploy facility
in resource frugal tiny edge devices, optimizing processing capacity, and improving reliability.
In this project, we present an application about such possibilities for TinyML in the context of
handwritten recognition challenge.
we propose the idea to make inference directly on a micro controller that is electronically linked with the input source of the character
and obtain the prediction of it. This is possible thanks to the Tiny ML technologies available
and in particular thanks to the TensorFlowLite library and its compatibility with the ESP32 microcontroller.

## Model

The model is based on the Lenet5 architecture and a modified version of it.

![LeNet5](https://user-images.githubusercontent.com/87801874/190124471-eca3ae8c-d509-4dda-850e-ece2d334c302.png)

In the modified version the last convolution is substitued with a Dense layer of 256.

## Application

We create an application that allows to take the input and collect the prediction of every single
letter in order to compose a word, shown on the display with a graphic interface,
with three buttons each one associated to an action. 

![application](https://user-images.githubusercontent.com/87801874/190124980-dba3ef93-efc1-4a76-9d0c-7d426ee6ef7d.jpg)
