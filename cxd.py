import laspy
from pyflann import *
import numpy as np
import math as m


def readAndTransformFile(filePath):
    file = laspy.file.File(filePath, mode="r")
    npFile = np.vstack([file.x, file.y, file.z]).transpose().astype('float64')
    npcFile = np.ascontiguousarray(npFile)

    return (npcFile, file)


def outputFileWithDist(outPath, outPoints, outHeader, distArray):
    outFile = laspy.file.File(outPath, mode="w", header=outHeader)
    outFile.points = outPoints
    outFile.intensity = distArray
    outFile.close()


def runLasFileDiff(fileOnePath, fileTwoPath):
    array1, lFile1 = readAndTransformFile(fileOnePath)
    array2, lFile2 = readAndTransformFile(fileTwoPath)

    distArray1 = processFlannIndices(array1, array2)
    distArray2 = processFlannIndices(array2, array1)

    outputFileWithDist(fileOnePath + ".new.las", lFile1.points, lFile1.header, distArray1)
    outputFileWithDist(fileTwoPath + ".new.las", lFile2.points, lFile2.header, distArray2)

    return


def processFlannIndices(primaryPts, neighborPts):

    distArray = np.array([], dtype='uint16')
    flann = FLANN()
    fParams = flann.build_index(primaryPts, target_precision=0.95, checks=-1)  # the "checks=-1" allows for all results to be returned

    ptIndex, ptDist = flann.nn(neighborPts, primaryPts, num_neighbors=1)
    ptSqrtDist = np.sqrt(ptDist) * 100
    ptSqrtDist.astype('uint16')

    return ptSqrtDist


if __name__ == "__main__":
    import argparse as ap

    file1 = "/home/davidbenjaminbrandt/PROJECTS/cxd/testData/treeSpeciesLarge.las"
    file2 = "/home/davidbenjaminbrandt/PROJECTS/cxd/testData/treeSpeciesLarge_new.las"

    runLasFileDiff(file1, file2)
