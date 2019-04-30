#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

' my image filter '

__author__ = 'Jing Tan'

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def myImageFilter(I, filter):
    filter = np.flipud(filter)
    filter = np.fliplr(filter)
    filter_sum = filter.sum()
    if filter_sum == 0:
        filter_sum = 1

    # get img url, filter vertical size, filter horizontal size and filter data,
    # img stored in I and filter stored in filter, filter size in filter_vsize and filter_hsize, filter data sum in filter_sum.
    k0 = int((filter.shape[0]-1)/2)
    k1 = int((filter.shape[1]-1)/2)
    I_zeropadded = np.zeros((I.shape[0]+ 2*k0, I.shape[1]+ 2*k1))
    I_zeropadded[k0:(k0+I.shape[0]), k1:(k1+I.shape[1])] = I
    # pad img with zeros

    I_result = np.zeros(I.shape)
    for i in range(0, I.shape[0]):
        for j in range(0, I.shape[1]):
            #I_zeropadded[i:i+filter.shape[0],j:j+filter.shape[1]]
            I_result[i][j] = np.multiply(filter, I_zeropadded[i:i+filter.shape[0],j:j+filter.shape[1]]).sum()
            I_result[i][j] = I_result[i][j]/filter_sum
    #I_result = I_filtered_zeropadding[k0:k0+I.shape[0],k1:k1+I.shape[1]]
    cv2.imwrite('./filtered_zero.jpg', I_result)
    return I_result
 
    #print(I_nearpadded
        #print(I_filtered_nearpadding)
#if __name__=='__main__':
#    myImageFilter()