import laspy
from pyflann import *
import numpy as np


def runLasFileDiff(fileOnePath, fileTwoPath):
    file1 = laspy.file.File(fileOnePath)
    file2 = laspy.file.File(fileTwoPath)

    
# Open a file in read mode:
inFile = laspy.file.File("./laspytest/data/simple.las")
# Grab a numpy dataset of our clustering dimensions:
dataset = np.vstack([inFile.X, inFile.Y, inFile.Z]).transpose()

# Find the nearest 5 neighbors of point 100.

neighbors = flann.nn(dataset, dataset[100,], num_neighbors = 5)
print("Five nearest neighbors of point 100: ")
print(neighbors[0])
print("Distances: ")
print(neighbors[1])

if __name__ == "__main__":
    import argparse as ap
