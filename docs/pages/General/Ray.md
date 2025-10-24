---
title: "Ray"
url: "https://docs.alliancecan.ca/wiki/Ray"
category: "General"
last_modified: "2023-03-13T12:22:59Z"
page_id: 22158
display_title: "Ray"
---

[Ray](https://docs.ray.io/) is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries for simplifying running parallel/distributed workloads, in particular Machine Learning jobs.

# Installation

## Latest available wheels {#latest_available_wheels}

To see the latest version of Ray that we have built:

For more information, see [Available wheels](https://docs.alliancecan.ca/Python#Available_wheels "Available wheels"){.wikilink}.

## Installing our wheel {#installing_our_wheel}

The preferred option is to install it using the Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install Ray in the virtual environment with `pip install`.

<!-- -->

:   

# Job submission {#job_submission}

## Single Node {#single_node}

Below is an example of a job that spawns a single-node Ray cluster with 6 cpus and 1 GPU.

In this simple example, we connect to the single-node Ray cluster launched in the job submission script, then we check that Ray sees the resources allocated to the job.

```{=mediawiki}
{{File
  |name=ray-example.py
  |lang="python"
  |contents=
import ray
import os

# Connect to Ray cluster
ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])

# Check that ray can see 6 cpus and 1 GPU
print(ray.available_resources())
}}
```
## Multiple Nodes {#multiple_nodes}

In the example that follows, we submit a job that spawns a two-node Ray cluster with 6 cpus and 1 GPU per node.

Where the script `config_env.sh` is:

And the script `launch_ray.sh` is:

```{=mediawiki}
{{File
  |name=launch_ray.sh
  |lang="bash"
  |contents=
#!/bin/bash

source $SLURM_TMPDIR/ENV/bin/activate
module load gcc/9.3.0 arrow

if [[ "$SLURM_PROCID" -eq "0" ]]; then
        echo "Ray head node already started..."
        sleep 10

else
        ray start --address "${HEAD_NODE}:${RAY_PORT}" --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=1 --block
        sleep 5
        echo "ray worker started!"
fi

}}
```
In this simple example, we connect to the two-node Ray cluster launched in the job submission script, then we check that Ray sees the resources allocated to the job.

```{=mediawiki}
{{File
  |name=test_ray.py
  |lang="python"
  |contents=
import ray
import os

# Connect to Ray cluster
ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])

# Check that Ray sees two nodes and their status is 'Alive'
print("Nodes in the Ray cluster:")
print(ray.nodes())

# Check that Ray sees 12 CPUs and 2 GPUs over 2 Nodes
print(ray.available_resources())
}}
```
# Hyperparameter search with Ray Tune {#hyperparameter_search_with_ray_tune}

Tune is a Ray module for experiment execution and hyperparameter tuning at any scale. It supports a wide range of frameworks including Pytorch, Tensorflow and Scikit-Learn. In the example that follows, we use Tune to perform a hyperparameter sweep and find the best combination of learning rate and batch size to train a convolutional neural network with Pytorch. You can find examples using other frameworks on [Ray\'s official documentation](https://docs.ray.io/en/latest/tune/examples/ml-frameworks.html)

To run this example, you can use one of the job submission templates provided [ above](https://docs.alliancecan.ca/#Job_submission " above"){.wikilink} depending on whether you require one or multiple nodes. As you will see in the code that follows, the amount of resources required by your job will depend mainly on two factors: the number of samples you wish to draw from the search space and the size of your model in memory. Knowing these two things you can reason about how many trials you will run in total and how many of them can run in parallel using as few resources as possible. For example, how many copies of your model can you fit inside the memory of a single GPU? That is the number of trials you can run in parallel using just one GPU.

In the example, our model takes up about 1GB in memory. We will run 20 trials in total, 10 in parallel at a time on the same GPU, and we will give one CPU to each trial to be used as a `DataLoader` worker. So we will pick the single node job submission template and we will replace the number of cpus per task with `#SBATCH --cpus-per-task=10` and the Python call with `python ray-tune-example.py --num_samples=20 --cpus-per-trial=1 gpus-per-trial=0.1`. We will also need to install the packages `ray[tune]` and `torchvision` in our virtualenv.

```{=mediawiki}
{{File
  |name=ray-tune-example.py
  |lang="python"
  |contents=

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import os

import argparse

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(config,num_workers):

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='/path/to/data', train=True, download=False, transform=transform)
    dataset_test = CIFAR10(root='/path/to/test_data', train=False, download=False, transform=transform)

    net = Net().cuda() # Load model on the GPU

    train_loader = DataLoader(dataset_train, batch_size=config["batch_size"], num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=config["batch_size"], num_workers=num_workers)

    criterion = nn.CrossEntropyLoss().cuda() # Load the loss function on the GPU
    optimizer = optim.SGD(net.parameters(), lr=config["lr"])

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total = 0
    correct = 0
    for batch_idx, (inputs, tagrets) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    session.report({"accuracy": correct / total})

parser = argparse.ArgumentParser(description='cifar10 hyperparameter sweep with ray tune')
parser.add_argument('--num_samples',type=int, default=10, help='')
parser.add_argument('--gpus_per_trial', type=float, default=1, help='')
parser.add_argument('--cpus_per_trial', type=int, default=1, help='')

def main():

    args = parser.parse_args()

    ## Connect to the Ray cluster launched in the job submission script
    ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])

    ## Define a search space for the sweep
    config = {
        "lr": tune.loguniform(1e-4, 1e-1), # candidate learning rates will be sampled from a log-uniform distrubution 
        "batch_size": tune.choice([2, 4, 8, 16]) # candidate batch sizes will be sampled randomly from this list of values
    }

 ## Our training loop only runs for one epoch. But if it ran for many epochs, a scheduler can kill trials before they end if they do not look promising
    scheduler = ASHAScheduler(
        max_t=1,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, num_workers=args.cpus_per_trial),
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial} # we set gpus_per_trial to 0.1, so each trial gets one tenth of a GPU
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
        param_space=config,
    )

    results = tuner.fit()

    best = results.get_best_result("accuracy","max")

    print("Best trial config: {}".format(best.config))

if __name__=='__main__':
   main()

}}
```
