---
title: "Migrating between clusters/en"
url: "https://docs.alliancecan.ca/wiki/Migrating_between_clusters/en"
category: "General"
last_modified: "2025-08-29T12:59:42Z"
page_id: 19746
display_title: "Migrating between clusters"
---

While our clusters have a relatively high degree of uniformity, particularly the general purpose clusters, there are still significant distinctions that you need to be aware of when moving from one cluster to another. This page explains what is the same between clusters and what changes you'll need to adapt to on a new cluster.

To transfer data from one cluster to another we recommend the use of Globus, particularly when the amount of data involved exceeds a few hundred megabytes.

=Access=

Each cluster is accessible via SSH, simply by changing the name of the cluster to the appropriate value; your username and password are the same across all of the clusters. Note that accessing Niagara does require a further step.

=Filesystems=

While each of the general purpose clusters has a similar  filesystem structure, it is important to realize that there is no mirroring of data between the clusters. The contents of your home, scratch and project spaces is independent on each cluster. The  quota policies may also differ between clusters though in general not by much. If a group you work with has a special storage allocation on one cluster, for example $HOME/projects/rrg-jsmith, it will normally only be available on that particular cluster. Equally so, if your group requested that the default project space quota on a cluster be increased from 1 to 10 TB, this change will only have been made on that cluster. To transfer the data from one cluster to another we recommend the use of Globus, particularly when the amount of data involved exceeds a few hundred megabytes.

=Software=

The collection of globally installed modules is the same across all of our general purpose clusters, distributed using CVMFS. For this reason, you should not notice substantial differences among the modules available assuming you are using the same  standard software environment. However, any Python virtual environments or R and Perl packages that you may have installed in your directories on one cluster will need to be re-installed on the new cluster, using the same steps that you employed on the original cluster. Equally so, if you modified your $HOME/.bashrc file on one cluster to customize your environment, you will need to modify the same file on the new cluster you're using. If you installed a particular program in your directories, this will also need to be re-installed on the new cluster since, as we mentioned above, the filesystems are independent between clusters.

=Job submission=

All of our clusters use Slurm for job submission, so many parts of a job submission script will work across clusters. However, you should note that the number of CPU cores per node or per GPU may significantly, so check the page of the cluster you are using to verify how many cores can be used on a node. The amount of memory per node or per core also varies, so you may need to adapt your script to account for this as well.

On some clusters, compute nodes may not have direct Internet access. Access the cluster's link, from the sidebar on the left, to find out about site specific policies.

Each research group has access to a default allocation on every cluster, e.g. #SBATCH --account=def-jsmith, however special compute allocations like RRG or contributed allocations are tied to a particular cluster and will not be available on other clusters.