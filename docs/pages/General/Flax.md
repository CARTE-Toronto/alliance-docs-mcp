---
title: "Flax"
url: "https://docs.alliancecan.ca/wiki/Flax"
category: "General"
last_modified: "2022-07-21T18:31:54Z"
page_id: 20509
display_title: "Flax"
---

[Flax](https://flax.readthedocs.io/en/latest/index.html) is a neural network library and ecosystem for [JAX](https://jax.readthedocs.io/en/latest/index.html) that is designed for flexibility. Its API for building models is similar to that of PyTorch and Keras where models are expressed as sequences of modules. Similarities however stop there - being based on JAX, Flax\'s API for training models is designed around functional programming.

# Installation

## Latest available wheels {#latest_available_wheels}

To see the latest version of Flax that we have built:

For more information, see [Available wheels](https://docs.alliancecan.ca/Python#Available_wheels "Available wheels"){.wikilink}.

## Installing the Compute Canada wheel {#installing_the_compute_canada_wheel}

The preferred option is to install it using the Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install Flax in the virtual environment with `pip install`.

<!-- -->

:   

# High Performance with Flax {#high_performance_with_flax}

## Flax with Multiple CPUs or a Single GPU {#flax_with_multiple_cpus_or_a_single_gpu}

As a framework based on JAX, Flax derives its high-performance from the combination of a functional paradigm, automatic differentiation and TensorFlow\'s [Accelerated Linear Algebra (XLA)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/index.md) compiler. Concretely, one can use JAX\'s Just-In-Time compiler to leverage XLA on code blocks (often compositions of functions) that are called repeatedly during a training loop, like loss computation, backpropagation and gradient updates. Another advantage this provides is that XLA handles compiling code blocks into CPU or GPU code transparently, so your Python code is exactly the same regardless of the device where it will be executed.

With the above being said, when training small scale models we strongly recommend using **multiple CPUs instead of using a GPU**. While training will almost certainly run faster on a GPU (except in cases where the model is very small), if your model and your dataset are not large enough, the speed up relative to CPU will likely not be very significant and your job will end up using only a small portion of the GPU\'s compute capabilities. This might not be an issue on your own workstation, but in a shared environment like our HPC clusters this means you are unnecessarily blocking a resource that another user may need to run actual large scale computations! Furthermore, you would be unnecessarily using up your group\'s allocation and affecting the priority of your colleagues\' jobs.

Simply put, **you should not ask for a GPU** if your code is not capable of making a reasonable use of its compute capacity. The following example illustrates how to submit a Flax job with or without a GPU:

```{=mediawiki}
{{File
  |name=flax-example.py
  |lang="python"
  |contents=

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import argparse
import time

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

def main():

   args = parser.parse_args()

   seed = jax.random.PRNGKey(42)

   class Net(nn.Module):

      @nn.compact
      def __call__(self,x):
         x = nn.Conv(features=6, kernel_size=(5, 5))(x)
         x = nn.relu(x)
         x = nn.max_pool(x, window_shape=(2, 2))
         x = nn.Conv(features=16, kernel_size=(5, 5))(x)
         x = nn.relu(x)
         x = nn.max_pool(x, window_shape=(2, 2))
         x = x.reshape((x.shape[0], -1))
         x = nn.Dense(features=120)(x)
         x = nn.relu(x)
         x = nn.Dense(features=84)(x)
         x = nn.relu(x)
         x = nn.Dense(features=10)(x)

         return x
   
   # Helper class to cast numpy arrays to JAX arrays
   class CastToJnp(object):
      def __call__(self, image):
         return np.array(image, dtype=jnp.float32)

   model = Net()

   params = model.init(seed, jnp.ones([3,32,32]))['params'] # initialize weights. Function takes in jnp.ones() of the same shape as the Model's inputs.
   optimizer = optax.sgd(args.lr)
   state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

   # Neither Flax or JAX provide pre-processing / data loading code. Here we use PyTorch for the job.

   transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), CastToJnp()])
   dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
   train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_jax)

   perf = []

   for batch_idx, (inputs, targets) in enumerate(train_loader):

      start = time.time()

      grads,loss = train_step(state, inputs, targets)
      state = update_state(state, grads)

      batch_time = time.time() - start
      images_per_sec = args.batch_size/batch_time
      perf.append(images_per_sec)

      print(f"Current Loss: {loss}")

   print(f"Images processed per second: {np.mean(perf)}")

# JIT Compile an entire training step: Forward and Backward in a single function call:
@jax.jit
def train_step(state, inputs, targets):

   def compute_loss(params):
      outputs = state.apply_fn({'params': params}, inputs)
      one_hot = jax.nn.one_hot(targets, 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits=outputs, labels=one_hot))
      return loss

   backward = jax.value_and_grad(compute_loss) # returns a function that autograds its input
   loss, grads = backward(state.params)
   return grads, loss

# JIT Compile weight update:
@jax.jit
def update_state(state, grads):
   return state.apply_gradients(grads=grads)

#Taken from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def collate_jax(batch):
   if isinstance(batch[0], np.ndarray):
      return np.stack(batch)
   elif isinstance(batch[0], (tuple,list)):
      transposed = zip(*batch)
      return [collate_jax(samples) for samples in transposed]
   else:
      return np.array(batch)

if __name__=='__main__':
   main()
}}
```
## Data Parallelism with Multiple GPUs {#data_parallelism_with_multiple_gpus}

Data Parallelism, in this context, refers to methods to perform training over multiple replicas of a model in parallel, where each replica receives a different chunk of training data at each iteration. Gradients are then aggregated at the end of an iteration and the parameters of all replicas are updated in a synchronous or asynchronous fashion, depending on the method. Using this approach may provide a significant speed-up by iterating through all examples in a large dataset approximately N times faster, where N is the number of model replicas. An important caveat of this approach, is that in order to get a trained model that is equivalent to the same model trained without Data Parallelism, the user must scale either the learning rate or the desired batch size in function of the number of replicas. See [this discussion](https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/13) for more information. In the examples that follow, each GPU hosts a replica of your model. Consequently, the model must be small enough to fit inside the memory of a single GPU.

### Single Node {#single_node}

```{=mediawiki}
{{File
  |name=flax-example-multigpu.py
  |lang="python"
  |contents=

from flax import linen as nn
from flax.training import train_state
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax

import time
import os
import functools

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

def main():

   args = parser.parse_args()

   seed = jax.random.PRNGKey(42)

   n_devices = jax.local_device_count() # get umber of GPUs available to the job

   class Net(nn.Module):

      @nn.compact
      def __call__(self,x):
         x = nn.Conv(features=6, kernel_size=(5, 5))(x)
         x = nn.relu(x)
         x = nn.max_pool(x, window_shape=(2, 2))
         x = nn.Conv(features=16, kernel_size=(5, 5))(x)
         x = nn.relu(x)
         x = nn.max_pool(x, window_shape=(2, 2))
         x = x.reshape((x.shape[0], -1))
         x = nn.Dense(features=120)(x)
         x = nn.relu(x)
         x = nn.Dense(features=84)(x)
         x = nn.relu(x)
         x = nn.Dense(features=10)(x)

         return x

   class CastToJnp(object):
      def __call__(self, image):
         return np.array(image, dtype=jnp.float32)

   model = Net()

   params = model.init(seed, jnp.ones([3,32,32]))['params']
   optimizer = optax.sgd(args.lr)
   state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

   state = jax_utils.replicate(state) # broadcast model replicas to all GPUs

   # Neither Flax or JAX provide pre-processing / data loading code. Here we use PyTorch for the job.

   transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), CastToJnp()])
   dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
   train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_jax)

   perf = []

   for batch_idx, (inputs, targets) in enumerate(train_loader):
       
      # Split a batch into "n_devices" sub-batches
      inputs = inputs.reshape(n_devices, inputs.shape[0] // n_devices, *inputs.shape[1:]) 
      targets = targets.reshape(n_devices, targets.shape[0] // n_devices, *targets.shape[1:])

      start = time.time()

      state, loss = train_step(state, inputs, targets)

      batch_time = time.time() - start
      images_per_sec = args.batch_size/batch_time
      perf.append(images_per_sec)

      print(f"Current Loss: {loss}")

   print(f"Images processed per second: {np.mean(perf)}")

# "jax.pmap" parallelizes inputs, function evaluation and outputs over a given axis. 
# This axis is first dimension of the inputs by default - it is the number of GPUs in this case.
# jax.map also JIT compiles the function, just like @jax.jit in the single GPU case. 

@functools.partial(jax.pmap, axis_name='gpus')
def train_step(state, inputs, targets):

   def compute_loss(params):
      outputs = state.apply_fn({'params': params}, inputs)
      one_hot = jax.nn.one_hot(targets, 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits=outputs, labels=one_hot))
      return loss

   backward = jax.value_and_grad(compute_loss)
   loss, grads = backward(state.params)

   grads = jax.lax.pmean(grads, axis_name='gpus') # compute the average of gradients of all model replicas
   loss = jax.lax.pmean(loss, axis_name='gpus')   # do the same with the loss

   updated_state = state.apply_gradients(grads=grads) # weight update is computed with averaged gradients from all replicas

   return updated_state, loss

def collate_jax(batch):
   if isinstance(batch[0], np.ndarray):
      return np.stack(batch)
   elif isinstance(batch[0], (tuple,list)):
      transposed = zip(*batch)
      return [collate_jax(samples) for samples in transposed]
   else:
      return np.array(batch)

if __name__=='__main__':
   main()
}}
```
### Multiple Nodes {#multiple_nodes}

```{=mediawiki}
{{File
  |name=flax-example-distributed.py
  |lang="python"
  |contents=
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax

import time
import os
import functools

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--coord_addr', type=str, default="127.0.0.1:34567",help='')

def main():

   args = parser.parse_args()

   world_size = int(os.environ.get("SLURM_NNODES"))
   rank = int(os.environ.get("SLURM_NODEID"))

   # Here we let jax know there's more than one node in this job
   jax.distributed.initialize(coordinator_address=args.coord_addr, num_processes=world_size, process_id=rank)

   n_devices = jax.local_device_count()

   seed = jax.random.PRNGKey(42)

   class Net(nn.Module):

         @nn.compact
         def __call__(self,x):
            x = nn.Conv(features=6, kernel_size=(5, 5))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2))
            x = nn.Conv(features=16, kernel_size=(5, 5))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(features=120)(x)
            x = nn.relu(x)
            x = nn.Dense(features=84)(x)
            x = nn.relu(x)
            x = nn.Dense(features=10)(x)

            return x

   class CastToJnp(object):
      def __call__(self, image):
         return np.array(image, dtype=jnp.float32)

   model = Net()

   params = model.init(seed, jnp.ones([3,32,32]))['params']
   optimizer = optax.sgd(args.lr)
   state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

   state = jax_utils.replicate(state) # Broadcast model to all GPUs on all nodes

   # Neither Flax or JAX provide pre-processing / data loading code. Here we use PyTorch for the job.
   # Here we add a DistributedSampler to make sure each node only sees a portion of the dataset

   transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), CastToJnp()])
   dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
   train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank)
   train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, collate_fn=collate_jax)

   perf = []

   for batch_idx, (inputs, targets) in enumerate(train_loader):
 
      # Split a batch into "n_devices" sub-batches. "n_devices" is the number of GPUs per node, not total GPUs!
      inputs = inputs.reshape(n_devices, inputs.shape[0] // n_devices, *inputs.shape[1:]) # 
      targets = targets.reshape(n_devices, targets.shape[0] // n_devices, *targets.shape[1:])

      start = time.time()

      state, loss = train_step(state, inputs, targets)

      batch_time = np.array(time.time() - start)

      if rank == 0:

         images_per_sec = args.batch_size / batch_time
         perf.append(images_per_sec)

         print(f"Current Loss: {loss}")

   if rank == 0:

      print(f"Images processed per second: {np.mean(perf)}")

# "jax.pmap" parallelizes inputs, function evaluation and outputs over a given axis. 
# This axis is the first dimension of the inputs by default - it is the number of local GPUs in this case.
# But because we called jax.distributed.initialize, computations will also be parallelized over nodes!
# jax.map also JIT compiles the function, just like @jax.jit in the single GPU case.

@functools.partial(jax.pmap, axis_name='gpus')
def train_step(state, inputs, targets):

   def compute_loss(params):
      outputs = state.apply_fn({'params': params}, inputs)
      one_hot = jax.nn.one_hot(targets, 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits=outputs, labels=one_hot))
      return loss

   backward = jax.value_and_grad(compute_loss)
   loss, grads = backward(state.params)

   grads = jax.lax.pmean(grads, axis_name='gpus') # compute the average of gradients of all model replicas across both GPUs and nodes
   loss = jax.lax.pmean(loss, axis_name='gpus') # do the same with loss

   updated_state = state.apply_gradients(grads=grads) # weight updates are computed using gradients averaged over all model replicas

   return updated_state, loss

def collate_jax(batch):
   if isinstance(batch[0], np.ndarray):
      return np.stack(batch)
   elif isinstance(batch[0], (tuple,list)):
      transposed = zip(*batch)
      return [collate_jax(samples) for samples in transposed]
   else:
      return np.array(batch)

if __name__=='__main__':
   main()

}}
```
