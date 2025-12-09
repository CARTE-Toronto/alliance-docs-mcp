---
title: "JupyterHub/en"
url: "https://docs.alliancecan.ca/wiki/JupyterHub/en"
category: "General"
last_modified: "2025-09-30T16:07:01Z"
page_id: 4743
display_title: "JupyterHub"
---

`<languages />`{=html} `<i>`{=html}JupyterHub is the best way to serve Jupyter Notebook for multiple users. It can be used in a class of students, a corporate data science group or scientific research group.`</i>`{=html} [^1]

JupyterHub provides a preconfigured version of JupyterLab and/or Jupyter Notebook; for more configuration options, please check the [Jupyter](https://docs.alliancecan.ca/Jupyter "Jupyter"){.wikilink} page.

# Alliance initiatives {#alliance_initiatives}

Some regional initiatives offer access to computing resources through JupyterHub.

## JupyterHub on clusters {#jupyterhub_on_clusters}

On the following clusters[^‡^](https://docs.alliancecan.ca/#clusters_note "‡"){.wikilink}, use your Alliance username and password to connect to JupyterHub:

  JupyterHub                                                                       Comments
  -------------------------------------------------------------------------------- -----------------------------------------------------------------------------------------------------------------------
  `<b>`{=html}[Béluga](https://jupyterhub.beluga.alliancecan.ca/)`</b>`{=html}     Provides access to JupyterLab servers spawned through jobs on the [Béluga](https://docs.alliancecan.ca/Béluga/en "Béluga"){.wikilink} cluster.
  `<b>`{=html}[Fir](https://jupyterhub.fir.alliancecan.ca/)`</b>`{=html}           Provides access to JupyterLab servers spawned through jobs on the [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink} cluster.
  `<b>`{=html}[Narval](https://jupyterhub.narval.alliancecan.ca/)`</b>`{=html}     Provides access to JupyterLab servers spawned through jobs on the [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink} cluster.
  `<b>`{=html}[Rorqual](https://jupyterhub.rorqual.alliancecan.ca/)`</b>`{=html}   Provides access to JupyterLab servers spawned through jobs on the [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink} cluster.

Some clusters provide access to JupyterLab through Open OnDemand. See [JupyterLab](https://docs.alliancecan.ca/JupyterLab "JupyterLab"){.wikilink} for more information.

`<b>`{=html}^‡^ Note that the compute nodes running the Jupyter kernels do not have internet access`</b>`{=html}. This means that you can only transfer files from/to your own computer; you cannot download code or data from the internet (e.g. cannot do \"git clone\", cannot do \"pip install\" if the wheel is absent from our [wheelhouse](https://docs.alliancecan.ca/Available_Python_wheels "wheelhouse"){.wikilink}). You may also have problems if your code performs downloads or uploads (e.g. in machine learning where downloading data from the code is often done).

## JupyterHub for universities and schools {#jupyterhub_for_universities_and_schools}

- The [Pacific Institute for the Mathematical Sciences](https://www.pims.math.ca) in collaboration with the Alliance and [Cybera](http://www.cybera.ca) offer cloud-based hubs to universities and schools. Each institution can have its own hub where users authenticate with their credentials from that institution. The hubs are hosted on Alliance [clouds](https://docs.alliancecan.ca/Cloud "clouds"){.wikilink} and are essentially for training purposes. Institutions interested in obtaining their own hub can visit [Syzygy](http://syzygy.ca).

# Server options {#server_options}

![`<i>`{=html}Server Options`</i>`{=html} form on Béluga\'s JupyterHub](https://docs.alliancecan.ca/JupyterHub_Server_Options.png "Server Options form on Béluga's JupyterHub") Once logged in, depending on the configuration of JupyterHub, the user\'s web browser is redirected to either `<b>`{=html}a)`</b>`{=html} a previously launched Jupyter server, `<b>`{=html}b)`</b>`{=html} a new Jupyter server with default options, or `<b>`{=html}c)`</b>`{=html} a form that allows a user to set different options for their Jupyter server before pressing the `<i>`{=html}Start`</i>`{=html} button. In all cases, it is equivalent to accessing requested resources via an [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink} on the corresponding cluster.

`<b>`{=html}Important:`</b>`{=html} On each cluster, only one interactive job at a time gets a priority increase in order to start in a few seconds or minutes. That includes `salloc`, `srun` and JupyterHub jobs. If you already have another interactive job running on the cluster hosting JupyterHub, your new Jupyter session may never start before the time limit of 5 minutes.

## Compute resources {#compute_resources}

For example, `<i>`{=html}Server Options`</i>`{=html} available on [Béluga\'s JupyterHub](https://jupyterhub.beluga.computecanada.ca/) are:

- `<i>`{=html}Account`</i>`{=html} to be used: any `def-*`, `rrg-*`, `rpp-*` or `ctb-*` account a user has access to
- `<i>`{=html}Time (hours)`</i>`{=html} required for the session
- `<i>`{=html}Number of (CPU) cores`</i>`{=html} that will be reserved on a single node
- `<i>`{=html}Memory (MB)`</i>`{=html} limit for the entire session
- (Optional) `<i>`{=html}GPU configuration`</i>`{=html}: at least one GPU
- `<i>`{=html}[User interface](https://docs.alliancecan.ca/JupyterHub#User_Interface "User interface"){.wikilink}`</i>`{=html} (see below)

## User interface {#user_interface}

While JupyterHub allows each user to use one Jupyter server at a time on each hub, there can be multiple options under `<i>`{=html}User interface`</i>`{=html}:

- `<b>`{=html}[JupyterLab](https://docs.alliancecan.ca/JupyterLab "JupyterLab"){.wikilink}`</b>`{=html} (modern interface): This is the most recommended Jupyter user interface for interactive prototyping and data visualization.
- Jupyter Notebook (classic interface): Even though it offers many functionalities, the community is moving towards [JupyterLab](https://docs.alliancecan.ca/JupyterHub#JupyterLab "JupyterLab"){.wikilink}, which is a better platform that offers many more features.
- Terminal (for a single terminal only): It gives access to a terminal connected to a remote account, which is comparable to connecting to a server through an SSH connection.

Note: JupyterHub could also have been configured to force a specific user interface. This is usually done for special events.

# JupyterLab

The JupyterLab interface is now described in our [JupyterLab](https://docs.alliancecan.ca/JupyterLab "JupyterLab"){.wikilink} page.

# Possible error messages {#possible_error_messages}

## Spawn failed: Timeout {#spawn_failed_timeout}

Most JupyterHub errors are caused by the underlying job scheduler which is either unresponsive or not able to find appropriate resources for your session. For example:

[thumb\|upright=1.1\|JupyterHub - Spawn failed: Timeout](https://docs.alliancecan.ca/File:JupyterHub_Spawn_failed_Timeout.png "thumb|upright=1.1|JupyterHub - Spawn failed: Timeout"){.wikilink}

- When starting a new session, JupyterHub automatically submits on your behalf a new [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink} to the cluster. If the job does not start within five minutes, a \"Timeout\" error message is raised and the session is cancelled.
  - Just like any interactive job on any cluster, a longer requested time can cause a longer wait time in the queue. Requesting a GPU or too many CPU cores can also cause a longer wait time. Make sure to request only the resources you need for your session.
  - If you already have another interactive job on the same cluster, your Jupyter session will be waiting along with other regular batch jobs in the queue. If possible, stop or cancel any other interactive job before using JupyterHub.
  - There may be just no resource available at the moment. Check the [status page](https://status.alliancecan.ca/) for any issue and try again later.

## Authentication error: Error 403 {#authentication_error_error_403}

Your account or your access to the cluster is currently inactive:

1.  Make sure your account is active, that is [`<b>`{=html}it has been renewed`</b>`{=html}](https://alliancecan.ca/en/services/advanced-research-computing/account-management/account-renewals)
2.  Make sure your [`<b>`{=html}access to a cluster`</b>`{=html}](https://ccdb.alliancecan.ca/me/access_services) is enabled

# References

[^1]: <http://jupyterhub.readthedocs.io/en/latest/index.html>
