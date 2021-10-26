# prior-localization
this is an old readme
Project seeking to localize encoding of the task prior in the brain

We aim to utilize GLMs and neural decoders to determine how much information a given neuron
or population of neurons encodes about various task-related variables. In particular we seek
to narrow down where we believe the task prior on target location is encoded during the IBL task.

This is heavily a work in progress, and relies on system calls to MATLAB via Python. Has not been
tested on any system besides Ubuntu 19.04, and likely will not work. Also requires a functional
MATLAB license to run.

Requires installation of contained Psytrack submodule (By Nick Roy & colleagues) via

```
cd ./psytrack
pip install -e .
```
