# Prior Localization
The repository provides the code associated with the manuscript 
[*Brain-wide representations of prior information in mouse decision-making*](https://doi.org/10.1101/2023.07.04.547684) (Findling, Hubert et al, 2023).

## Dependencies
The code has been tested on Ubuntu 20.04 and 22.04, Rocky Linux 8.8 and OSX 13.4.1, using Python 3.8, 3.9 and 3.10.
Required Python software packages are listed in [requirements.txt](https://github.com/int-brain-lab/prior-localization/blob/main/requirements.txt). 

## Installation
The installation takes about 7 min on a standard desktop computer. It is recommended to set up and activate a clean environment using conda or virtualenv, e.g.
```shell
virtualenv prior --python=python3.10
source prior/bin/activate
```

Then clone this repository and install it along with its dependencies
```shell
git clone https://github.com/int-brain-lab/prior-localization.git
cd prior-localization
pip install .
```

In a Python console, test if you can import functions from prior_localization
```python
from prior_localization.functions.decoding import fit_session_ephys
```


## Connecting to IBL database
In order to run the example code or the tests, you need to connect to the public IBL database to access example data.
Our API, the Open Neurophysiology Environment (ONE) has already been installed with the requirements. 
If you have never used ONE, you can just establish the default database connection like this in a Python console. 
The first time you instantiate ONE you will have to enter the password (`international`) 
```python
from one.api import ONE
ONE.setup(silent=True)
one = ONE()
```

**NOTE**: if you have previously used ONE with a different database you might want to run this instead. Again, the 
first time you instantiate ONE you will have to enter the password (`international`)
```python
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', make_default=False, silent=True)
one = ONE(base_url='https://openalyx.internationalbrainlab.org')
```

If you run into any issues refer to the [ONE documentation](https://int-brain-lab.github.io/ONE/index.html)

## Running example code
We provide an example script in 
[prior_localization/decode_single_session.ipynb](https://github.com/int-brain-lab/prior-localization/blob/main/prior_localization/decode_single_session.ipynb) 
that performs a region-level 
decoding of the Bayes optimal prior from pre-stimulus neural activity. The data is
automatically downloaded from the public IBL database, provided that the above ONE setup has been performed.


This script has been tested on a laptop computer (Intel® Core™ i7 processor, 4 cores, 16GB RAM) running Ubuntu 22.04, with Python 3.10 and Python package versions listed in 
[software_versions_example.txt](https://github.com/int-brain-lab/prior-localization/blob/main/software_versions_example.txt).
In this setup, it takes about 2 min to run (including data download times).

## Running tests
To run the full set of tests you can use e.g. unittest
```shell
python -m unittest discover -s prior_localization/tests
```

## Code Description
The code heavily relies on the fitting functions in `prior_localication/functions/decoding.py`. 
The inputs of this function are described in the function's description and in the tutorial file.
This codes outputs the paths of the folders which contain the decoding results. See the tutorial file
to plot output predictions of the decoding results.
