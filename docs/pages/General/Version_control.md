---
title: "Version control/en"
url: "https://docs.alliancecan.ca/wiki/Version_control/en"
category: "General"
last_modified: "2023-11-22T20:47:54Z"
page_id: 236
display_title: "Version control"
---

`<languages />`{=html}

## Introduction

Source code management is one of the cornerstones of application development. When the time comes to manage the source code of a project you\'re working on, there are two ways to proceed. You could make multiple backup copies, send the source code to your colleagues by email and waste time trying to remember who has or is using which version of the code and how to reconcile the modifications that each contributor has made. Alternatively, you could choose a much more rational approach by using a revision control system that has been specifically created to make this process as painless as possible.

All significant applications and libraries are developed using such tools. For academic research, these tools are even more important because traceability is essential for ensuring that a given set of results can be reproduced. A good metaphor is that revision control management tools are the programmer\'s equivalent to the experimentalist\'s lab notebook.

## Advantages

Revision control tools offer you a great many advantages and these more than compensate for the occasional inconvenience. Firstly, they permit you to collaborate more easily. They eliminate the risk that a collaborator might delete your modifications or vice versa without leaving a trace. These tools save the history of all the modifications made to a project and in that way function somewhat like a time machine, allowing you to reinitialize your project to an earlier version, in order to reproduce your results for example. They also make it easier to document these changes, so that all of the users of a project are notified of the changes made and the reasons they were made.

## Basic functionality {#basic_functionality}

Source code management tools function using a basic principle of separating local modifications made by a user, in his or her local directory, and what is called the repository. The repository contains, in a structured manner, the history of all of the modifications made by all of a project\'s contributors. The development of a software project using a source code management tool is thus modified in comparison to the development of a purely `<i>`{=html}local`</i>`{=html} project. Rather than simply saving your modifications to the local disk drive, you as a contributor have to submit (`<i>`{=html}commit`</i>`{=html}) your modifications to the repository in order to make them available to other developers. Inversely, developers need to make sure they are using the latest version of a file by retrieving it from the repository (`<i>`{=html}checkout`</i>`{=html}, `<i>`{=html}update`</i>`{=html}) before making their own modifications. If two programmers modify the same source code file at the same time, the source code management tool may report a conflict during the submission of the two rival modifications or automatically resolve the conflict if possible.

## Families of revision control tools {#families_of_revision_control_tools}

The principal revision control tools can be divided into two `<i>`{=html}families`</i>`{=html} or `<i>`{=html}generations`</i>`{=html}. First-generation tools, which include [CVS](https://en.wikipedia.org/wiki/Concurrent_Versions_System) and [SVN](https://en.wikipedia.org/wiki/Apache_Subversion), use a single central repository. All of the modifications are under the control of and retrieved from this authoritative repository. Second-generation tools such as [Git](https://en.wikipedia.org/wiki/Git_%28software%29), [Mercurial](https://en.wikipedia.org/wiki/Mercurial) and [Bazaar](https://en.wikipedia.org/wiki/GNU_Bazaar), use local repositories. The advantage of such a system is that development work can be done independent of any remote server. The `<i>`{=html}commit`</i>`{=html} and `<i>`{=html}checkout`</i>`{=html} operations can thus be much faster and perform more complex operations. As an example, Git and Mercurial offer advanced management of branched development. In a branched development model, each new feature corresponds to a branch of the development tree. The `<i>`{=html}production`</i>`{=html} version of the project then corresponds to the principal branch and additional features are developed in parallel until they are sufficiently mature to be fused with the principal branch or else abandoned and the branch left to die. This development model is particularly well adapted to very large projects involving several programmers.

In exchange for this flexibility, with second-generation tools all modifications that need to go to an external repository must do so in two steps: first, they are submitted to the local repository (`<i>`{=html}commit`</i>`{=html}), then they are pushed (`<i>`{=html}push`</i>`{=html}) to the external repository. Equally so, to retrieve the modifications from an external repository, you must first obtain them (`<i>`{=html}pull`</i>`{=html} or `<i>`{=html}get`</i>`{=html}) so they can be imported into the local repository, then update your working version (`<i>`{=html}update`</i>`{=html} or `<i>`{=html}checkout`</i>`{=html}).

## Choosing a tool {#choosing_a_tool}

If you want to contribute to an existing project, you don\'t really have any choice; you will have to use the tool that has been chosen by the initial development team. If you are starting your own project, the choice will depend on the breadth of your project. If it\'s a project with only a few contributors, which will remain private and for which you would simply like to have a history of all the modifications, a first-generation tool like [SVN](https://en.wikipedia.org/wiki/Apache_Subversion) can be sufficient. If your project is larger, with external collaborators, you should consider a second-generation tool like [Git](https://en.wikipedia.org/wiki/Git_%28software%29) or [Mercurial](https://en.wikipedia.org/wiki/Mercurial).

### Repository hosting {#repository_hosting}

Another question to consider when choosing a version control tool is where you will host your repository. If you and your collaborators are always working on the same single machine then having a local repository only visible on that machine could be sufficient. However, if you are working across multiple machines, or working with collaborators working on different machines, a repository accessible via the internet will be helpful. This will allow you to easily synchronize your code between machines and also provide additional safety for your code by being distributed. There are a number of ways to accomplish this, from hosting and setting up the repository yourself on your own server (e.g. [svn](https://civicactions.com/blog/how-to-set-up-an-svn-repository-in-7-simple-steps/),[git](https://git-scm.com/book/en/v2/Git-on-the-Server-The-Protocols),[gitlab](https://about.gitlab.com/?utm_source=google&utm_medium=cpc&utm_campaign=Search%20-%20Brand&utm_content=GitLab%20-%20Open%20Source%20Git&utm_term=gitlab&gclid=CPWslub9vtACFZSEaQodwzoAew), [gitbucket](https://github.com/gitbucket/gitbucket)), to using one of the available online services (e.g. [bitbucket](https://bitbucket.org/product), [github](https://github.com/), [gitlab](https://about.gitlab.com/), [sourceforge](https://sourceforge.net/)) which is hosted on their servers and do not require you to have a server that is always accessible.

## See also {#see_also}

See [here](https://www.youtube.com/watch?v=EmMNIMDl9hM) for a very short video demonstrating the basics of version control with Git.
