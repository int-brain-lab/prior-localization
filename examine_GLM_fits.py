from iofuns import loadmat
import numpy as np
import matplotlib.pyplot as plt


currfits = loadmat('./fullfitout_frontalcort1.mat')
currfits = currfits['ans']