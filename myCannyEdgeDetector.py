from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import filter
import grayscale

class SEdgel:
	''' Edge class represents an edge detected.'''
	def __init__(self, midpoint, endpoint1, endpoint2, strength, theta, n):
		'''init with midpoints and endpoints'''
		self.endpoints = [endpoint1, endpoint2]
		self.x = midpoint[0] # sub-pixel edge position x (midpoint)
		self.y = midpoint[1] # sub-pixel edge position y (midpoint)
		self.strength = strength # strength of edgel (gradient magnitude)
		self.n = n # orientation, as normal vector
		self.angle = theta #orientation, as angle (degrees)
		self.length = 1 # length of edgel ?
		self.isMarked = 0

def getmidpoint(endpoint1, endpoint2):
	S1 = abs(S[endpoint1[0]][endpoint1[1]])
	S2 = abs(S[endpoint2[0]][endpoint2[1]])
	x = (endpoint1[0]*S1 + endpoint2[0]*S2)/(S1+S2)
	y = (endpoint1[1]*S1 + endpoint2[1]*S2)/(S1+S2)
	return (x,y)

def getEverything(endpoint1, endpoint2):
	'''Compute gradient along with edge strength, normal vector and angles.'''
	S1 = S[endpoint1[0]][endpoint1[1]]
	S2 = S[endpoint2[0]][endpoint2[1]]
	if (endpoint1[0] == endpoint2[0]):
		Gx = 0
		Gy = S2 - S1
		theta = 270
	else:
		Gy = 0
		Gx = S2 - S1
		theta = 0
	strength = np.sqrt(Gx ** 2+Gy ** 2)
	n = (Gx, Gy)
	return strength, theta, n

def getKernel(s,k):
	#s, k = 1, 2 #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
	probs = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)] 
	kernel = np.outer(probs, probs)
	return kernel

I = cv2.imread("Lenna.png")
I = grayscale.gray(I)
''' 1. Blur the picture '''
# picture filter module I wrote.
#kernel = getKernel(0.6,2)
#kernel_k = getKernel(0.72,2)
#B = filter.myImageFilter(I, kernel)
#B_k = filter.myImageFilter(I, kernel_k)
#B_doG = (B_k-B)/0.12

kernel_dog = np.array([\
	0,1,1,2,2,2,1,1,0,\
	1,2,4,5,5,5,4,2,1,\
	1,4,5,3,0,3,5,4,1,\
	2,5,3,-12,-24,-12,3,5,2,\
	2,5,0,-24,-40,-24,0,5,2,\
	2,5,3,-12,-24,-12,3,5,2,\
	1,4,5,3,0,3,5,4,1,\
	1,2,4,5,5,5,4,2,1,\
	0,1,1,2,2,2,1,1,0])
kernel_dog = kernel_dog.reshape((9,9))
B_doG = filter.myImageFilter(I, kernel_dog)
#B = gaussian_filter(I, sigma=1)
cv2.imwrite('Lenna_doG.jpg', B_doG)

S = B_doG

''' 4. count zero crosses in a 2x2 window; 5. 6. 7. 8.'''
Edges = []
for i in range(I.shape[0]-1):
	for j in range(I.shape[1]-1):
		zeroCross = []
		if S[i][j]*S[i+1][j] < 0:
			zeroCross.append(1) # Horizontal
		else:
			zeroCross.append(0)
		if S[i][j]*S[i][j+1] < 0:
			zeroCross.append(1) # Vertical
		else:
			zeroCross.append(0)
		if S[i][j+1]*S[i+1][j+1] < 0:
			zeroCross.append(1) # Horizontal
		else:
			zeroCross.append(0)
		if S[i+1][j]*S[i+1][j+1] < 0:
			zeroCross.append(1) # Vertical
		else:
			zeroCross.append(0)
		if sum(zeroCross) == 2:
			if zeroCross[0] == 1:
				mid = getmidpoint((i,j),(i+1,j))
				strength, theta, n = getEverything((i,j),(i+1,j))
				edge = SEdgel(mid, (i,j), (i+1,j), strength, theta, n)
				Edges.append(edge)
			if zeroCross[1] == 1:
				mid = getmidpoint((i,j),(i,j+1))
				strength, theta, n = getEverything((i,j),(i,j+1))
				edge = SEdgel(mid, (i,j), (i,j+1), strength, theta, n)
				Edges.append(edge)
			if zeroCross[2] == 1:
				mid = getmidpoint((i,j+1),(i+1,j+1))
				strength, theta, n = getEverything((i,j+1),(i+1,j+1))
				edge = SEdgel(mid, (i,j+1), (i+1,j+1), strength, theta, n)
				Edges.append(edge)
			if zeroCross[3] == 1:
				mid = getmidpoint((i+1,j),(i+1,j+1))
				strength, theta, n = getEverything((i+1,j),(i+1,j+1))
				edge = SEdgel(mid, (i+1,j), (i+1,j+1), strength, theta, n)
				Edges.append(edge)
print("Detected edge suspect number: ", len(Edges))
'''Extra: Thresholding'''
hThreshold = 1550
lThreshold = -400
Thresholding = np.zeros(I.shape)
sum = 0
Strength = np.zeros(I.shape)
for i in Edges:
	x = int(i.x)
	y = int(i.y)
	Strength[x][y] = i.strength
	if i.strength > hThreshold:
		Thresholding[x][y] = 1
		sum = sum + 1
	else:
		if i.strength > lThreshold:
			Thresholding[x][y] = 2
print("Detected edge number first round:", sum)
cv2.imwrite("Lenna_strength.jpg", Strength)
sum = 0
for i in range(Thresholding.shape[0]-1):
	for j in range(Thresholding.shape[1]-1):
		 if (Thresholding[i][j] == 2):
		 	sum = sum + 1
		 	if Thresholding[i-1][j]==1 or Thresholding[i-1][j-1] == 1 \
		 	 or Thresholding[i+1][j]==1  or Thresholding[i+1][j+1]==1 \
		 	 or Thresholding[i][j-1]==1  or Thresholding[i+1][j-1]==1 \
		 	 or Thresholding[i-1][j+1]==1 or Thresholding[i][j+1]==1 :
		 	 Thresholding[i][j] == 1
		 	else:
		 		Thresholding[i][j] == 0
print("Detected middle strength points:", sum)
'''Final Outcome'''
Result = np.zeros(I.shape)
sum = 0
for i in range(Thresholding.shape[0]):
	for j in range(Thresholding.shape[1]):
		if Thresholding[i][j] == 1:
			sum = sum + 1
			Result[i][j] = 255
	#sum = sum + i.strength
	#Result[x][y] = 255
	#endpoints = i.endpoints
	#for p in endpoints:
	#	Result[p[0]][p[1]] = 255
#print("Average strength", sum/len(Edges))
print("Detected edge number: ", sum)
cv2.imwrite("result.jpg",Result)






