---
title: "Ray"
url: "https://docs.alliancecan.ca/wiki/Ray"
category: "General"
last_modified: "2025-11-14T21:28:51Z"
page_id: 22158
display_title: "Ray"
---

[Ray](https://docs.ray.io/) is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries for simplifying running parallel/distributed workloads, in particular Machine Learning jobs.

# Installation

## Latest available wheels {#latest_available_wheels}

To see the latest version of Ray that we have built:

For more information, see [Available wheels](https://docs.alliancecan.ca/Python#Available_wheels "wikilink").

## Installing our wheel {#installing_our_wheel}

The preferred option is to install it using the Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "wikilink"), thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "wikilink").
:   3\. Install Ray in the virtual environment with `pip install`.

```{=html}
<!-- -->
```

:   

# Job submission {#job_submission}

In the example that follows, we submit a job that spawns a single-node Ray cluster with 200GB RAM, 48 CPU cores and 4 H100 GPUs per node.

The script can be used for multi-node Ray clusters by changing the default value set by the line `#SBATCH --nodes=1`, or overriding it with `sbatch`\'s `--nodes=` command-line argument.

```{=mediawiki}
{{File
  |name=ray-example.sh
  |lang="bash"
  |contents=
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out

# 1. Load modules
module load python

# 2. & 3. Create a virtualenv and install Ray on all nodes
#         Then, activate that virtualenv in this shell.
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh
source $SLURM_TMPDIR/ENV/bin/activate

# Define Ray head node address (this node) & port (any free port will do)
export HEAD_NODE=$(hostname)
export RAY_PORT=$(
    python -c 'import socket;             \
               s=socket.socket();         \
               s.bind(("", 0));           \
               print(s.getsockname()[1]); \
               s.close()'
)

# Craft --multi-prog configuration for Ray
ray_multiprog_config="$SLURM_TMPDIR/ray-multiprog.conf"
cat <<EOF >"$ray_multiprog_config"
0   ray start --block --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT
*   ray start --block --address=${HEAD_NODE}:${RAY_PORT}
EOF

#
# Some older Ray versions malfunctioned if the worker nodes started before
# the head node. In that case, please modify the worker-node launch configuration
# above to add a 10-second delay, using the following alternative lines:
#
# cat <<EOF >"$ray_multiprog_config"
# 0   ray start --block --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT
# *   sh -c 'sleep 10; exec "\$@"' sh ray start --block --address=${HEAD_NODE}:${RAY_PORT}
# EOF
#

# Launch head and workers on any/all nodes allocated to the job.
srun -N $SLURM_NNODES -n $SLURM_NNODES                       \
     --multi-prog "$ray_multiprog_config"                    \
     ${SLURM_CPUS_ON_NODE:+--num-cpus} ${SLURM_CPUS_ON_NODE} \
     ${SLURM_GPUS_ON_NODE:+--num-gpus} ${SLURM_GPUS_ON_NODE} &
ray_cluster_pid=$!

# Run your own script here
python test_ray.py "$@"
sleep 10

# Shut down Ray worker nodes after the Python script exits.
# Await its exit.
kill $ray_cluster_pid
wait

# Stage out your results, if any
# cp $SLURM_TMPDIR/results.json ~/scratch/

}}
```
Where the script `config_env.sh` is:

In this simple example, we connect to the Ray cluster launched in the job submission script, then we check that Ray sees the resources allocated to the job.

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

To run this example, you can use one of the job submission templates provided [ above](https://docs.alliancecan.ca/#Job_submission "wikilink") depending on whether you require one or multiple nodes. As you will see in the code that follows, the amount of resources required by your job will depend mainly on two factors: the number of samples you wish to draw from the search space and the size of your model in memory. Knowing these two things you can reason about how many trials you will run in total and how many of them can run in parallel using as few resources as possible. For example, how many copies of your model can you fit inside the memory of a single GPU? That is the number of trials you can run in parallel using just one GPU.

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
