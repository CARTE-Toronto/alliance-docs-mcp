---
title: "Optuna"
url: "https://docs.alliancecan.ca/wiki/Optuna"
category: "General"
last_modified: "2022-01-14T16:03:58Z"
page_id: 18344
display_title: "Optuna"
---

[Optuna](https://optuna.org/) is an automatic hyperparameter optimization (HPO) software framework, particularly designed for machine learning.

Please refer to the [Optuna Documentation](https://optuna.readthedocs.io/en/stable/) for a definition of terms, tutorial, API, etc.

## Using Optuna on Compute Canada {#using_optuna_on_compute_canada}

Here is a sketch of an SBATCH script for an HPO using Optuna:

```{=mediawiki}
{{File
  |name=hpo_with_optuna.sh
  |lang="sh"
  |contents=
#!/bin/bash 
#SBATCH -A def-account
#SBATCH --array 1-N%M   # This will launch N jobs, but only allow M to run in parallel
#SBATCH --time TIME     # Each of the N jobs will have the time limit defined in here.
... other SBATCH arguments ...

# Each trial in the study will be run in a separate job.
# The Optuna study_name has to be set to be able to continue an existing study.
OPTUNA_STUDY_NAME=my_optuna_study1

# Specify a path in your home, or on project.
OPTUNA_DB=$HOME/${OPTUNA_STUDY_NAME}.db

# Launch your script, giving it as arguments the database file and the study name
python train.py --optuna-db $OPTUNA_DB --optuna-study-name $OPTUNA_STUDY_NAME
}}
```
It\'s important for `M` to be much smaller than `N`, to let the optimization process do its thing. At the limit, if all trials execute simultaneously, they won\'t benefit from \"past knowledge\", and it will be equivalent to doing a random search. As for evolution and natural selection, there has to be a sequence of generations.

In `train.py`, create and launch the Optuna study like the following. For the rest of the code, see the [Optuna Documentation](https://optuna.readthedocs.io/en/stable/).

`# args.optuna_db and args.optuna_study_name are command line arguments`\
\
`study = optuna.create_study(`\
`    storage='sqlite:///' + args.optuna_db,`\
`    study_name=args.optuna_study_name,`\
`    load_if_exists=True`\
`)`\
`...`\
`study.optimize(objective, n_trials=1)  # Only execute a single trial at a time, to avoid wasting compute`

Remember that we are launching a separate job for each trial. Thus, we want our python script to stop after a single trial. Else, a subsequent trial will start, and the job will be killed while it\'s running.
