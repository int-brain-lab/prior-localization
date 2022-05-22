# Delphi
The repo provides code to run encoding and decoding models for the entire IBL 
brainwide map dataset.

We aim to utilize GLMs and neural decoders to determine how much information a given neuron or 
population of neurons encodes about various task- and behavior-related variables.
This code has been optimized to run many parallel encoding/decoding jobs on a cluster.

This is heavily a work in progress.

## Installation

First create a Conda environment in which this package and its dependencies will be 
installed.

```console 
foo@bar:~$ conda create --name delphi
```

and activate it:

```console
foo@bar:~$ conda activate delphi
```

Move into the folder where you want to place the repository folder, 
and then download it from GitHub:

```console
foo@bar:~$ cd <SOME_FOLDER>
foo@bar:~$ git clone https://github.com/int-brain-lab/prior-localization.git
```

Then move into the newly-created repository folder, and install dependencies:

```console
foo@bar:~$ cd prior-localization
foo@bar:~$ pip install -e .
```

## Download data and store in a local cache
TODO

## Run encoding models
TODO

## Run decoding models
TODO
