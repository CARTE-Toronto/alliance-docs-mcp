---
title: "Keras/en"
url: "https://docs.alliancecan.ca/wiki/Keras/en"
category: "General"
last_modified: "2023-06-27T16:13:02Z"
page_id: 9908
display_title: "Keras"
---

`<languages />`{=html}

\"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.\"[^1]

If you are porting a Keras program to one of our clusters, you should follow [our tutorial on the subject](https://docs.alliancecan.ca/Tutoriel_Apprentissage_machine/en "our tutorial on the subject"){.wikilink}.

## Installing

1.  Install [TensorFlow](https://docs.alliancecan.ca/TensorFlow "TensorFlow"){.wikilink}, CNTK, or Theano in a Python [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
2.  Activate the Python virtual environment (named `$HOME/tensorflow` in our example).

    :   
3.  Install Keras in your virtual environment.

    :   

### R package {#r_package}

This section details how to install Keras for R and use TensorFlow as the backend.

1.  Install TensorFlow for R by following [ these instructions](https://docs.alliancecan.ca/Tensorflow#R_package " these instructions"){.wikilink}.
2.  Follow the instructions from the parent section.
3.  Load the required modules.

    :   
4.  Launch R.

    :   
5.  In R, install the Keras package with `devtools`.
    :

devtools::install_github(\'rstudio/keras\')

</syntaxhighlight>

You are then good to go. Do not call `install_keras()` in R, as Keras and TensorFlow have already been installed in your virtual environment with `pip`. To use the Keras package installed in your virtual environment, enter the following commands in R after the environment has been activated.

``` r
library(keras)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))
```

## References

[^1]: <https://keras.io/>
