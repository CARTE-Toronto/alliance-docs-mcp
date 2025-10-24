---
title: "Deepspeed"
url: "https://docs.alliancecan.ca/wiki/Deepspeed"
category: "General"
last_modified: "2025-07-18T15:12:27Z"
page_id: 23434
display_title: "Deepspeed"
---

DeepSpeed is a deep learning training optimization library, providing the means to train massive billion parameter models at scale. Fully compatible with PyTorch, DeepSpeed features implementations of novel memory-efficient distributed training methods, based on the Zero Redundancy Optimizer (ZeRO) concept. Through the use of ZeRO, DeepSpeed enables distributed storage and computing of different elements of a training task - such as optimizer states, model weights, model gradients and model activations - across multiple devices, including GPU, CPU, local hard disk, and/or combinations of these devices. This \"pooling\" of resources, notably for storage, allows models with massive amounts of parameters to be trained efficiently, across multiple nodes, without explicitly handling Model, Pipeline or Data Parallelism in your code.

## Installing Deepspeed {#installing_deepspeed}

Our recommendation is to install it using our provided Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install both PyTorch and Deepspeed in the virtual environment with `pip install`.

<!-- -->

:   

## Multi-GPU and multi-node jobs with Deepspeed {#multi_gpu_and_multi_node_jobs_with_deepspeed}

In the example that follows, we use `deepspeed` to reproduce our [PyTorch tutorial](https://docs.alliancecan.ca/PyTorch#PyTorch_with_Multiple_GPUs "PyTorch tutorial"){.wikilink} on how to train a model with multiple GPUs distributed over multiple nodes. Notable differences are:

:   1\. Here we define and configure several common elements of the training task (such as optimizer, learning rate scheduler, batch size and more) in a config file, rather than using code in the main python script.
:   2\. We also define Deepspeed specific configurations, such as what modality of ZeRO to utilize, in a config file.

Where the script `config_env.sh` is:

The script `launch_training_deepseed.sh` is as shown below. Notice that we use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) to launch our python script. While Deepspeed has [its own launcher](https://pytorch.org/docs/stable/elastic/run.html), we do not recommend using it at this time:

Next we define and configure our training task in the file `ds_config.json`. Here we setup ZeRO stage 0, meaning ZerRO is disabled - no model parallelism will take place and this will be a purely data parallel job. We also enable mixed-precision training, where some tensors are computed/stored in half-precision (fp16) to accelerate computations using up less memory space. See [Deepspeed\'s documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.runtime.zero.config.DeepSpeedZeroConfig) for more details on all configurable parameters.

```{=mediawiki}
{{File
  |name=ds_config.json
  |lang="json"
  |contents=
{
  "train_batch_size": 16,
  "steps_per_print": 2000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 1000
    }
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "fp16": {
      "enabled": true,
      "fp16_master_weights_and_grads": false,
      "loss_scale": 0,
      "loss_scale_window": 500,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 15
  },
  "wall_clock_breakdown": false,
  "zero_optimization": {
      "stage": 0,
      "allgather_partitions": true,
      "reduce_scatter": true,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": true,
      "contiguous_gradients": true,
      "cpu_offload": false
  }
}

}}
```
And finally, `pytorch-deepspeed.py` is:

## Using PyTorch Lightning {#using_pytorch_lightning}

In the following tutorial, we use PyTorch Lightning as a wrapper around Deepspeed and demonstrate how to use ZeRO Stage 3 with a pool of GPUs, with offloading to the CPU, and with offloading to the compute node\'s local storage.

### ZeRO on GPU {#zero_on_gpu}

In the following example, we use ZeRO Stage 3 to train a model using a \"pool\" of 4 GPUs. Stage 3 means all three of: optimizer states; model parameters; and model gradients will be split (sharded) between all 4 GPUs. This is more memory-efficient than pure Data Parallelism, where we would have a full replica of the model loaded on each GPU. Using DeepSpeed\'s optimizer `FusedAdam` instead of a native PyTorch one, performance is comparable with pure Data Parallelism. DeepSpeed\'s optimizers are JIT compiled at run-time and you must load the module `cuda/``<version>`{=html} where **`<version>`{=html}** must match the version used to build the PyTorch install you are using.

### ZeRO with offload to CPU {#zero_with_offload_to_cpu}

In this example, we will again use ZeRO stage 3, but this time we enable offloading model parameters and optimizers states to the CPU. This means that the compute node\'s memory will be available to store these tensors while they are not required by any GPU computations, and additionally, optimizer steps will be computed on the CPU. For practical purposes, you can think of this as though your GPUs were gaining an extra 32GB of memory. This takes even more pressure off from GPU memory and would allow you to increase your batch size, for example, or increase the size of the model. Using DeepSpeed\'s optimizer `DeepSpeedCPUAdam` instead of a native PyTorch one, performance remains at par with pure Data Parallelism. DeepSpeed\'s optimizers are JIT compiled at run-time and you must load the module `cuda/``<version>`{=html} where **`<version>`{=html}** must match the version used to build the PyTorch install you are using.

### ZeRO with offload to NVMe {#zero_with_offload_to_nvme}

In this example, we use ZeRO stage 3 yet again, but this time we enable offloading model parameters and optimizers states to the local disk. This means that the compute node\'s local disk storage will be available to store these tensors while they are not required by any GPU computations. As before, optimizer steps will be computed on the CPU. Again, for practical purposes, you can think of this as extending GPU memory by however much storage is available on the local disk, though this time performance will significantly degrade. This approach works best (i.e., performance degradation is least noticeable) on NVMe-enabled drives, which have higher throughput and faster response times, but it can be used with any type of storage.

```{=mediawiki}
{{File
  |name=deepspeed-stage3-offload-nvme.py
  |lang="python"
  |contents=
import os

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.strategies import DeepSpeedStrategy

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, deepspeed offload to nvme test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--max_epochs', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

def main():
    print("Starting...")

args = parser.parse_args()

class ConvPart(nn.Module):

def __init__(self):
          super(ConvPart, self).__init__()

self.conv1 = nn.Conv2d(3, 6, 5)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.relu = nn.ReLU()

def forward(self, x):
          x = self.pool(self.relu(self.conv1(x)))
          x = self.pool(self.relu(self.conv2(x)))
          x = x.view(-1, 16 * 5 * 5)

return x

# Dense feedforward part of the model
    class MLPPart(nn.Module):

def __init__(self):
          super(MLPPart, self).__init__()

self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)
          self.relu = nn.ReLU()

def forward(self, x):
          x = self.relu(self.fc1(x))
          x = self.relu(self.fc2(x))
          x = self.fc3(x)

return x

class Net(pl.LightningModule):

def __init__(self):
          super(Net, self).__init__()

self.conv_part = ConvPart()
          self.mlp_part = MLPPart()

def configure_sharded_model(self):

self.block = nn.Sequential(self.conv_part, self.mlp_part)

def forward(self, x):
          x = self.block(x)

return x

def training_step(self, batch, batch_idx):
          x, y = batch
          y_hat = self(x)
          loss = F.cross_entropy(y_hat, y)
          return loss

def configure_optimizers(self):
          return DeepSpeedCPUAdam(self.parameters())

net = Net()

""" Here we initialize a Trainer() explicitly with 1 node and 2 GPU.
        To make this script more generic, you can use torch.cuda.device_count() to set the number of GPUs
        and you can use int(os.environ.get("SLURM_JOB_NUM_NODES")) to set the number of nodes. 
        We also set progress_bar_refresh_rate=0 to avoid writing a progress bar to the logs, 
        which can cause issues due to updating logs too frequently."""

local_scratch = os.environ['SLURM_TMPDIR'] # Get path where local storage is mounted

print(f'Offloading to: {local_scratch}')

trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy=DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        remote_device="nvme",
        offload_params_device="nvme",
        offload_optimizer_device="nvme",
        nvme_path="local_scratch",
        ), max_epochs = args.max_epochs)

transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

trainer.fit(net,train_loader)

if __name__=='__main__':
   main()

}}
```
