# -*- coding: utf-8 -*-

from skimage import feature
import numpy as np
import cv2

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="default")
        #cv2.imshow('LBP', lbp)
        #cv2.waitKey(0)

        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numPoints + 3),range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        #cv2.imshow('Histogram', hist)
        #cv2.waitKey(0)
        # return the histogram of Local Binary Pattern
        return hist


if __name__=="__main__":
    desc = LocalBinaryPatterns(24, 8)
    image = cv2.imread("/Users/Vivien/Desktop/test.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
