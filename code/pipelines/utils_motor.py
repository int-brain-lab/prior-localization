# old imports
from one.api import ONE
from brainbox.io.one import load_channel_locations 
import brainbox.behavior.wheel as wh
from brainbox.processing import bincount2D
from ibllib.atlas import regions_from_allen_csv
import ibllib.atlas as atlas
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader
import numpy as np
from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy
import pandas as pd
import random
import seaborn as sns
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.stats import zscore
import itertools
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from scipy.interpolate import interp1d
import matplotlib
from scipy import stats
from scipy.stats import percentileofscore