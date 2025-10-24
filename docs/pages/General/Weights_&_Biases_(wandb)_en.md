---
title: "Weights & Biases (wandb)/en"
url: "https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb)/en"
category: "General"
last_modified: "2025-08-31T14:15:55Z"
page_id: 15989
display_title: "Weights & Biases (wandb)"
---

`<languages />`{=html}

[Weights & Biases (wandb)](https://wandb.ai) is a `<i>`{=html}meta machine learning platform`</i>`{=html} designed to help AI practitioners and teams build reliable machine learning models for real-world applications by streamlining the machine learning model lifecycle. By using wandb, you can track, compare, explain and reproduce machine learning experiments.

## Using wandb on Alliance clusters {#using_wandb_on_alliance_clusters}

### Availability on compute nodes {#availability_on_compute_nodes}

Full usage of `wandb` on compute nodes requires internet access as well as access to Google Cloud Storage, both of which may not be available depending on the cluster:

+-----------+--------------------+--------------------------------------------------------------------+
| Cluster   | Wandb Availability | Note                                                               |
+===========+====================+====================================================================+
| Narval    | Limited ❌         | Users from **MILA and other eligible groups only** via `httpproxy` |
+-----------+                    |                                                                    |
| Rorqual   |                    |                                                                    |
+-----------+                    |                                                                    |
| TamIA     |                    |                                                                    |
+-----------+--------------------+--------------------------------------------------------------------+
| Fir       | Yes ✅             | `httpproxy` not required                                           |
+-----------+--------------------+--------------------------------------------------------------------+
| Nibi      | Yes ✅             | `httpproxy` not required                                           |
+-----------+--------------------+--------------------------------------------------------------------+
| Trillium  | No ❌              | internet access is disabled on compute nodes                       |
+-----------+--------------------+--------------------------------------------------------------------+
| Vulcan    | Yes ✅             | `httpproxy` not required                                           |
+-----------+--------------------+--------------------------------------------------------------------+
| Killarney | Yes ✅             | `httpproxy` not required                                           |
+-----------+--------------------+--------------------------------------------------------------------+
|           |                    |                                                                    |
+-----------+--------------------+--------------------------------------------------------------------+

## Users from MILA and other eligible groups {#users_from_mila_and_other_eligible_groups}

Members of the MILA Québec AI Institute may use `wandb` on any of our clusters with internet access, provided that they use a valid **Mila-org** Weights & Biases account to log into `wandb`. Please see the table above for more information on modules required for using `wandb` on each cluster.

Other groups are known to have made arrangements with Weights & Biases to bypass calls to the Google Cloud Storage API. Please contact your PI to find out if your group has made such arrangements.

## Narval, Rorqual and TamIA {#narval_rorqual_and_tamia}

While it is possible to upload basic metrics to Weights&Biases during a job on Narval, Rorqual and TamIA, the wandb package will automatically attempt to upload information about your environment to a Google Cloud Storage bucket, which is not allowed on the compute nodes of these clusters. This will result in a crash during or at the very end of a training run. Your job may also freeze until it reaches its wall time, thereby wasting resources. It is not currently possible to disable this behaviour. Note that uploading artifacts to W&B with `wandb.save()` also requires access to Google Cloud Storage and will cause your job to freeze or crash.

You can still use wandb by enabling the [`offline`](https://docs.wandb.ai/library/cli#wandb-offline) mode. In this mode, wandb will write all metrics, logs and artifacts to the local disk and will not attempt to sync anything to the Weights&Biases service on the internet. After your jobs finish running, you can sync their wandb content to the online service by running the command [`wandb sync`](https://docs.wandb.ai/ref/cli#wandb-sync) on the login node.

Note that [Comet.ml](https://docs.alliancecan.ca/Comet.ml "Comet.ml"){.wikilink} is a product very similar to Weights & Biases, and works on Narval, Rorqual and TamIA.

## Example

The following is an example of how to use wandb to track experiments in offline mode. To run in online mode, load the module `httpproxy` on applicable clusters and follow the comments on the example script below.

The script wandb-test.py is a simple example of metric logging. See [W&B\'s full documentation](https://docs.wandb.ai) for more options.

```{=mediawiki}
{{File
  |name=wandb-test.py
  |lang="python"
  |contents=
import wandb

wandb.init(project="wandb-pytorch-test", settings=wandb.Settings(start_method="fork"))

for my_metric in range(10):
    wandb.log({'my_metric': my_metric})

}}
```
After a training run in offline mode, there will be a new folder `./wandb/offline-run*`. You can send the metrics to the server using the command `wandb sync ./wandb/offline-run*`. Note that using `*` will sync all runs.
