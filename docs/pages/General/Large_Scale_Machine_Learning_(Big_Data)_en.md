---
title: "Large Scale Machine Learning (Big Data)/en"
url: "https://docs.alliancecan.ca/wiki/Large_Scale_Machine_Learning_(Big_Data)/en"
category: "General"
last_modified: "2023-11-28T20:44:25Z"
page_id: 19194
display_title: "Large Scale Machine Learning (Big Data)"
---

`<languages />`{=html}

In the field of Deep Learning, the widespread use of mini-batching strategies along with first-order iterative solvers makes most common training tasks naturally scalable to large quantities of data. Whether you are looking at training Deep Neural Networks on a few thousand examples, or hundreds of millions of them, the flow of your code will look pretty much the same: load a few examples from a target source (from disk, from memory, from a remote source\...) and iterate through them, computing gradients and using them to update the parameters of the model as you go. Conversely, in many Traditional Machine Learning packages \--notably `scikit-learn`\-- scaling your code to train on very large datasets is often not trivial. Many algorithms that fit common models such as Generalized Linear Models (GLMs) and Support Vector Machines (SVMs) for example, may have default implementations that require the entire training set to be loaded in memory and often do not leverage any manner of thread or process parallelism. Some of these implementations also rely on memory-intensive solvers, which may require several times the size of your training set\'s worth of memory to work properly.

This page covers options to scale out traditional machine learning methods to very large datasets. Whether your training workload is too massive to fit even in a large memory node, or just big enough to take a really long time to process serially, the sections that follow may provide some insights to help you train models on Big Data.

# Scikit-learn {#scikit_learn}

[Scikit-learn](https://scikit-learn.org/stable/index.html) is a Python module for machine learning that is built on top of SciPy and distributed under the 3-Clause BSD license. This popular package features an intuitive API that makes building fairly complex machine learning pipelines very straightforward. However, many of its implementations of common methods such as GLMs and SVMs assume that the entire training set can be loaded in memory, which might be a showstopper when dealing with massive datasets. Furthermore, some of these algorithms opt for memory-intensive solvers by default. In some cases, you can avoid these limitations using the ideas that follow.

## Stochastic gradient solvers {#stochastic_gradient_solvers}

If your training set is small enough that it can be loaded entirely in memory, but you are experiencing Out-Of-Memory (OOM) errors during training, the culprit is likely a memory-intensive solver. Many common machine learning methods in `scikit-learn` have variations of [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) available as an option and replacing the default solver by an SGD-based one is often a straightforward solution to OOM errors.

The following example compares a [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) performed using the default solver with an SGD-based one. You can monitor memory usage by running the command `htop` on the terminal while the Python code runs.

Another option that reduces memory usage even more, is to use [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) instead of Ridge. This class implements many types of generalized linear models for regression, using a vanilla stochastic gradient descent as a solver. One caveat of using SGDRegressor is that it only works if the output is unidimensional (a scalar).

## Batch learning {#batch_learning}

In cases where your dataset is too large to fit in memory \--or just large enough that it does not leave enough memory free for training\-- it is possible to leave your data on disk and load it in batches during training, similar to how deep learning packages work. Scikit-learn refers to this as [`<i>`{=html}out-of-core learning`</i>`{=html}](https://scikit-learn.org/stable/computing/scaling_strategies.html) and it is a viable option whenever an estimator has the `partial_fit` [method available](https://scikit-learn.org/stable/computing/scaling_strategies.html?highlight=partial_fit#incremental-learning). In the examples below, we perform out-of-core learning by iterating over datasets stored on disk.

In this first example, we use [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) to fit a linear SVM classifier with batches of data coming from a pair of `<b>`{=html}numpy`</b>`{=html} arrays. These arrays are stored on disk as [npy files](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#npy-format) and we will keep them there by [memory-mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) these files. Since `SGDClassifier` has the `partial_fit` method, we can iterate through our large memory-mapped files loading only a small batch of rows from the arrays in memory at a time. Each call to `partial_fit` will then run one epoch of stochastic gradient descent over a batch of data.

Another common method of storing data for Machine Learning is using CSV files. In this example, we train a LASSO regression model reading data in batches from a CSV file using the [pandas](https://pandas.pydata.org) package.

# Snap ML {#snap_ml}

[Snap ML](https://snapml.readthedocs.io/en/latest/) is a closed-source machine learning library developed by IBM that currently supports a number of classical machine learning models and scales gracefully to datasets with billions of examples and/or features. It offers distributed training, GPU acceleration and supports sparse data structures. It features an API very similar to scikit-learn and can be used as a replacement for that library when dealing with massive datasets.

## Installation

### Latest available wheels {#latest_available_wheels}

To see the latest version of Snap ML that we have built:

For more information, see [Available wheels](https://docs.alliancecan.ca/Python#Available_wheels "Available wheels"){.wikilink}.

### Installing the wheel {#installing_the_wheel}

The preferred option is to install it using the Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install SnapML in the virtual environment with `pip install`.

<!-- -->

:   

## Multithreading

All estimators in Snap ML support thread parallelism, which can be controlled via the `n_jobs` parameter. Setting this parameter to the number of cores available in your job will typically deliver a good speedup relative to the scikit-learn implementation of the same estimator. The following is a performance comparison of `Ridge` between scikit-learn and Snap ML.

```{=mediawiki}
{{File
  |name=ridge-snap-vs-sklearn.py
  |lang="python"
  |contents=
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from snapml import LinearRegression
import time

X,y = make_regression(n_samples=100000, n_features=10000, n_informative=50)

model_sk = Ridge(solver='saga')

print("Running Ridge with sklearn...")
tik = time.perf_counter()
model_sk.fit(X,y)
tok = time.perf_counter()

print(f"sklearn took {tok-tik:0.2f} seconds to fit.")

model_snap = LinearRegression(penalty='l2',n_jobs=4)

print("Running Ridge with SnapML...")
tik = time.perf_counter()
model_snap.fit(X,y)
tok = time.perf_counter()

print(f"SnapML took {tok-tik:0.2f} seconds to fit.")
}}
```
## Training on GPU {#training_on_gpu}

All estimators in `Snap ML` support GPU acceleration, with one or multiple GPUs. For single GPU training, simply set the parameter `use_gpu=True`. For multiple GPU training, in addition to setting `use_gpu`, pass a list containing the GPU IDs available to your job to `device_ids`. For example, inside a job that requested 2 GPUs, set `device_ids=[0,1]` to use both GPUs for training. The following example extends the performance comparison from the previous section to include training on GPU with Snap ML, this time training an SVM classifier with a non-linear kernel.

```{=mediawiki}
{{File
  |name=ridge-snap-vs-sklearn2.py
  |lang="python"
  |contents=
from sklearn.datasets import make_classification
from sklearn.svm import SVC 
from snapml import SupportVectorMachine
import time

X,y = make_classification(n_samples=100000, n_features=10000, n_classes=3, n_informative=50)

model_sk = SVC(kernel='rbf') #sklearn's SVM fit-time scales at least quadratically with the number of samples... this will take a loooong time.

print("Running SVM Classifier with sklearn...")
tik = time.perf_counter()
model_sk.fit(X,y)
tok = time.perf_counter()

print(f"sklearn took {tok-tik:0.2f} seconds to fit.")

model_snap = SupportVectorMachine(kernel='rbf',n_jobs=4)

print("Running SVM Classifier with SnapML without GPU...")
tik = time.perf_counter()
model_snap.fit(X,y)
tok = time.perf_counter()

print(f"SnapML took {tok-tik:0.2f} seconds to fit without GPU.")

model_snap_gpu = SupportVectorMachine(kernel='rbf',n_jobs=4, use_gpu=True)

print("Running SVM Classifier with SnapML with GPU...")
tik = time.perf_counter()
model_snap_gpu.fit(X,y)
tok = time.perf_counter()

print(f"SnapML took {tok-tik:0.2f} seconds to fit with GPU.")
}}
```
## Out-of-memory training {#out_of_memory_training}

All estimators in Snap ML use first-order iterative solvers, similar to SGD, by default. It is thus possible to perform training in batches and avoid loading entire datasets in memory. Unlike scikit-learn however, Snap ML accepts memory-mapped numpy arrays as inputs directly.

## MPI

Snap ML features distributed implementations of many estimators. To run in distributed mode, call your Python script using `mpirun` or `srun`.

# Spark ML {#spark_ml}

[Spark ML](https://spark.apache.org/docs/latest/ml-guide.html) is a machine learning library built on top of [Apache Spark](https://docs.alliancecan.ca/Apache_Spark/en "Apache Spark"){.wikilink}. It enables users to scale out many machine learning methods to massive amounts of data, over multiple nodes, without worrying about distributing datasets or explicitly writing distributed/parallel code. The library also includes many useful tools for distributed linear algebra and statistics. Please see our tutorial on [submitting Spark jobs](https://docs.alliancecan.ca/Apache_Spark/en#Usage "submitting Spark jobs"){.wikilink} before trying out the examples on the official [Spark ML documentation](https://spark.apache.org/docs/latest/ml-guide.html).
