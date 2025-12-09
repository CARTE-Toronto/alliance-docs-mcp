---
title: "Flax"
url: "https://docs.alliancecan.ca/wiki/Flax"
category: "General"
last_modified: "2022-07-21T18:31:54Z"
page_id: 20509
display_title: "Flax"
---

Flax is a neural network library and ecosystem for JAX that is designed for flexibility. Its API for building models is similar to that of PyTorch and Keras where models are expressed as sequences of modules. Similarities however stop there - being based on JAX, Flax's API for training models is designed around functional programming.

= Installation =

==Latest available wheels==
To see the latest version of Flax that we have built:

For more information, see Available wheels.

==Installing the Compute Canada wheel==

The preferred option is to install it using the Python wheel as follows:
:1. Load a Python module, thus module load python
:2. Create and start a virtual environment.
:3. Install Flax in the virtual environment with pip install.

:

= High Performance with Flax =

== Flax with Multiple CPUs or a Single GPU ==

As a framework based on JAX, Flax derives its high-performance from the combination of a functional paradigm, automatic differentiation and TensorFlow's Accelerated Linear Algebra (XLA) compiler. Concretely, one can use JAX's Just-In-Time compiler to leverage XLA on code blocks (often compositions of functions) that are called repeatedly during a training loop, like loss computation, backpropagation and gradient updates. Another advantage this provides is that XLA handles compiling code blocks into CPU or GPU code transparently, so your Python code is exactly the same regardless of the device where it will be executed.

With the above being said, when training small scale models we strongly recommend using multiple CPUs instead of using a GPU. While training will almost certainly run faster on a GPU (except in cases where the model is very small), if your model and your dataset are not large enough, the speed up relative to CPU will likely not be very significant and your job will end up using only a small portion of the GPU's compute capabilities. This might not be an issue on your own workstation, but in a shared environment like our HPC clusters this means you are unnecessarily blocking a resource that another user may need to run actual large scale computations! Furthermore, you would be unnecessarily using up your group's allocation and affecting the priority of your colleagues' jobs.

Simply put, you should not ask for a GPU if your code is not capable of making a reasonable use of its compute capacity. The following example illustrates how to submit a Flax job with or without a GPU:

== Data Parallelism with Multiple GPUs ==
Data Parallelism, in this context, refers to methods to perform training over multiple replicas of a model in parallel, where each replica receives a different chunk of training data at each iteration. Gradients are then aggregated at the end of an iteration and the parameters of all replicas are updated in a synchronous or asynchronous fashion, depending on the method. Using this approach may provide a significant speed-up by iterating through all examples in a large dataset approximately N times faster, where N is the number of model replicas. An important caveat of this approach, is that in order to get a trained model that is equivalent to the same model trained without Data Parallelism, the user must scale either the learning rate or the desired batch size in function of the number of replicas. See this discussion for more information. In the examples that follow, each GPU hosts a replica of your model. Consequently, the model must be small enough to fit inside the memory of a single GPU.

=== Single Node ===

=== Multiple Nodes ===