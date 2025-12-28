---
title: "Classiq"
url: "https://docs.alliancecan.ca/wiki/Classiq"
category: "General"
last_modified: "2025-12-16T16:39:37Z"
page_id: 32094
display_title: "Classiq"
---

Classiq is quantum computing software that lets you design, optimize and execute quantum programs and is free to use for academic researchers (sign up on classiq.io with an institutional email).

The software can be used via their online platform, platform.classiq.io. Execution of quantum programs designed and optimized with Classiq can be run on multiple provider backends. Classiq also provides a Python SDK that can be used on other platforms.

NOTE: While Classiq is largely hardware independent for execution, synthesizing (compiling) quantum programs with Classiq requires use of the Classiq platform via their API and requires authentication. Therefore, even if using local execution, synthesis can only be done on compute nodes with internet access. As such, Fir and Nibi are suitable choices for using Classiq.

=Installation=

Installation on Alliance clusters is straightforward. Classiq will work with all versions of Python >= 3.11:

module load python/3.11 scipy-stack/2025a
virtualenv --no-download ~/ENV && source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index classiq

=Authentication on Alliance Clusters=

Before using the Python SDK with Classiq, you will need to authenticate with Classiq on one of the login nodes. This requires first signing up for an account with Classiq at platform.classiq.io. In order to obtain the free academic use license, please sign up using your institutional email.

1. Sign up for an account with Classiq at platform.classiq.io

2. Proceed with installation of the Python SDK using the instructions above.

3. Activate the virtual environment in which you installed Classiq and start a python terminal.

(ENV) [user@server ~]$ python
Python 3.11.5 (main, Sep 19 2023, 16:07:22) [GCC 12.3.1 20230526] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>

4. Import the authenticate module and run the authentication, this will open a browser window, press "q" and confirm with "Y" to exit the browser:

>>> from classiq import authenticate
>>> authenticate()

5. After quitting the browser, you will see a message, giving a URL to the authentication website. You can copy and paste this website into a browser on your own computer and sign into Classiq to verify:

>>> authenticate()
If a browser doesn't automatically open, please visit this URL from any trusted device to authenticate:
Your user code:

6. Verify the user code given on the Classiq website matches that shown in the terminal on the cluster and proceed with authentication.