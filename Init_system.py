''' Create a galaxy of N bodies using diferent models'''
from common import *
from numpy import loadtxt


Data = loadtxt('Parameters', dtype=str)

# Number of bodies.
N = int(Data[2])

# Initial radius of the distribution
ini_radius = float(Data[3]) #kpc

# Model of Galaxy
Mod = Data[6]

# Folder to save the data
data_folder = Data[10]

# Format of files
format = Data[11]

# Create system
system_init_write(N, read_model(Mod), ini_radius, format, data_folder)
