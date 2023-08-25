# Prior Localization
The repo provides code to run prior decoding for the entire IBL brainwide map dataset.

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
If you have never used ONE, you can just establish the default database connection like this in a Python console: 
```python
from one.api import ONE
ONE.setup(silent=True)
one = ONE()
```

**NOTE**: if you have previously used ONE you might want to either skip the previous step or set up a new connection 
to the public IBL database (accept defaults for usernames and passwords):
```python
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org')
```

If you run into any issues refer to the [ONE documentation](https://int-brain-lab.github.io/ONE/index.html)

## Running example code
We provide an example script that performs ... . The data is
automatically downloaded from the public IBL database, provided that the above ONE setup has been performed.


This script has been tested on a laptop computer (Intel® Core™ i7 processor, 4 cores, 16GB RAM) running Ubuntu 22.04, with Python 3.10 and Python package versions listed in 
[software_versions_example.txt](https://github.com/int-brain-lab/prior-localization/blob/main/software_versions_example.txt).
In this setup, it takes about 2 min to run (including data download times).

## Running tests
To run the full set of tests you can use e.g. unittest
```shell
python -m unittest discover -s prior_localization/tests
```
