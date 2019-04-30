#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

' gray '

__author__ = 'Jing Tan'

import cv2
def gray(I) :
	image = cv2.imread('./lenna.png')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('./gray.jpg', gray)
	return gray

#if __name__=='__main__':
#    gray()

