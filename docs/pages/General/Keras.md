---
title: "Keras/en"
url: "https://docs.alliancecan.ca/wiki/Keras/en"
category: "General"
last_modified: "2023-06-27T16:13:02Z"
page_id: 9908
display_title: "Keras"
---

"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano."https://keras.io/

If you are porting a Keras program to one of our clusters, you should follow our tutorial on the subject.

==Installing==

#Install TensorFlow, CNTK, or Theano in a Python virtual environment.
#Activate the Python virtual environment (named $HOME/tensorflow in our example).
#:
#Install Keras in your virtual environment.
#:

=== R package ===

This section details how to install Keras for R and use TensorFlow as the backend.

#Install TensorFlow for R by following  these instructions.
#Follow the instructions from the parent section.
#Load the required modules.
#:
# Launch R.
#:
#In R, install the Keras package with devtools.
#:
devtools::install_github('rstudio/keras')

You are then good to go. Do not call install_keras() in R, as Keras and TensorFlow have already been installed in your virtual environment with pip. To use the Keras package installed in your virtual environment, enter the following commands in R after the environment has been activated.

library(keras)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))

== References ==