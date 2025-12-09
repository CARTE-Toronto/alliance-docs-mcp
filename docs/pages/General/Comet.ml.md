---
title: "Comet.ml/en"
url: "https://docs.alliancecan.ca/wiki/Comet.ml/en"
category: "General"
last_modified: "2025-08-29T15:40:47Z"
page_id: 12614
display_title: "Comet.ml"
---

`<languages />`{=html}

[Comet](https://comet.ml) is a meta machine learning platform designed to help AI practitioners and teams build reliable machine learning models for real-world applications by streamlining the machine learning model lifecycle. By using Comet, users can track, compare, explain and reproduce their machine learning experiments. Comet can also greatly accelerate hyperparameter search, by providing a [module for the Bayesian exploration of hyperparameter space](https://www.comet.ml/parameter-optimization).

## Using Comet on our clusters {#using_comet_on_our_clusters}

### Availability

Since it requires an internet connection, Comet has restricted availability on compute nodes, depending on the cluster:

+-----------+--------------------+----------------------------------------------+
| Cluster   | Wandb Availability | Note                                         |
+===========+====================+==============================================+
| Narval    | Yes ✅             | `module load httpproxy` required             |
+-----------+                    |                                              |
| Rorqual   |                    |                                              |
+-----------+                    |                                              |
| TamIA     |                    |                                              |
+-----------+--------------------+----------------------------------------------+
| Fir       | Yes ✅             | `httpproxy` not required                     |
+-----------+--------------------+----------------------------------------------+
| Nibi      | Yes ✅             | `httpproxy` not required                     |
+-----------+--------------------+----------------------------------------------+
| Trillium  | No ❌              | internet access is disabled on compute nodes |
+-----------+--------------------+----------------------------------------------+
| Vulcan    | Yes ✅             | `httpproxy` not required                     |
+-----------+--------------------+----------------------------------------------+
| Killarney | Yes ✅             | `httpproxy` not required                     |
+-----------+--------------------+----------------------------------------------+
|           |                    |                                              |
+-----------+--------------------+----------------------------------------------+

### Best practices {#best_practices}

- Avoid logging metrics (e.g. loss, accuracy) at a high frequency. This can cause Comet to throttle your experiment, which can make your job duration harder to predict. As a rule of thumb, please log metrics (or request new hyperparameters) at an interval \>= 1 minute.
