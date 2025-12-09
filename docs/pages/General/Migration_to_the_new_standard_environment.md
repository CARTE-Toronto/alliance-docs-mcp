---
title: "Migration to the new standard environment/en"
url: "https://docs.alliancecan.ca/wiki/Migration_to_the_new_standard_environment/en"
category: "General"
last_modified: "2024-02-14T15:21:35Z"
page_id: 16321
display_title: "Migration to the new standard environment"
---

= What are the differences between StdEnv/2023 and the earlier environments? =
The differences are discussed in Standard software environments.

= Can I change my default standard environment? =
After 2024 April 1,  StdEnv/2023 will be the default environment for all clusters. However, you can specify your own default environment at any time by modifying the $HOME/.modulerc file.  For example, running the following command will set your default environment to StdEnv/2020:

You must log out and log in again for this change to take effect.

= Do I need to reinstall/recompile my code when the StdEnv changes? =
Yes. If you compile your own code, or have installed R or Python packages, you should recompile your code or reinstall the packages you need with the newest version of the standard environment.

= How can I use an earlier environment? =
If you have an existing workflow and want to continue to use the same software versions you are using now, simply add
  module load StdEnv/2020
to your job scripts before loading any other modules.

= Will the earlier environments be removed? =
The earlier environments and any software dependent on them will remain available, but versions 2016.4 and 2018.3 are no longer supported, and we recommend not using them. Our staff will only install software in the new environment 2023.

= Can I mix modules from different environments? =
No, you should use a single environment for a given job - different jobs can use different standard environments by explicitly
loading one or the other at the job's beginning but within a single job you should only use a single environment. The results of trying to mix different environments are unpredictable but in general will lead to errors of one kind or another.

= Which environment should I use? =
If you are starting a new project, or if you want to use a newer version of an application, you should use StdEnv/2023 by adding
  module load StdEnv/2023
to your job scripts. This command does not need to be deleted to use StdEnv/2023 after April 1.

= Can I keep using an older environment by loading modules in my .bashrc? =
Loading modules in your .bashrc is not recommended. Instead, explicitly load modules in your job scripts.

= I don't use the HPC clusters but cloud resources only. Do I need to worry about this? =
No, this change will only affect the Available software accessed by  using environment modules.

= I can no longer load a module that I previously used =
More recent versions of most applications are installed in the new environment. To see the available versions, run the module avail command. For example,

shows several versions of the GCC compilers, which may be different from those in earlier environments.