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

def DownSampling(I, filter):
	''' down sample rate is 2'''
	filter = np.flipud(filter)
	filter = np.fliplr(filter)
	filter_sum = filter.sum()
	if filter_sum == 0:
		filter_sum = 1
	k0 = int((filter.shape[0]-1)/2)
	k1 = int((filter.shape[1]-1)/2)
	I_zeropadded = np.zeros((I.shape[0]+ 2*k0, I.shape[1]+ 2*k1))
	I_zeropadded[k0:(k0+I.shape[0]), k1:(k1+I.shape[1])] = I
	Result = np.zeros((int(I.shape[0]/2), int(I.shape[1]/2)))
	for i in range(0, int(I.shape[0]/2)):
		for j in range(0, int(I.shape[1]/2)):
			Result[i][j] = np.multiply(filter, I_zeropadded[i*2:i*2+filter.shape[0],j*2:j*2+filter.shape[1]]).sum()
			Result[i][j] = Result[i][j]/filter_sum
	return Result

def UpSampling(I):
	filter = np.array([6,1,0,0,4,4,0,0,1,6,1,0,0,4,4,0,0,1,6,1,0,0,4,4,0,0,1,6,0,0,0,4])
	filter = filter.reshape((4, 8))
	print(filter)
	Result = np.zeros((I.shape[0]*2, I.shape[1]*2))
	I_addzero = np.zeros((I.shape[0]*2, I.shape[1]*2))
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			I_addzero[2*i][2*j] = I[i][j]
	filter = np.flipud(filter)
	filter = np.fliplr(filter)
	filter_sum = filter.sum()/2
	if filter_sum == 0:
		filter_sum = 1
	k0 = int((filter.shape[0])/2)
	k1 = int((filter.shape[1])/2)
	I_zeropadded = np.zeros((I_addzero.shape[0]+ 2*k0, I_addzero.shape[1]+ 2*k1))
	I_zeropadded[k0:(k0+I_addzero.shape[0]), k1:(k1+I_addzero.shape[1])] = I_addzero
	I_result = np.zeros(I_addzero.shape)
	for i in range(0, I_addzero.shape[0]):
		for j in range(0, I_addzero.shape[1]):
			I_result[i][j] = np.multiply(filter, I_zeropadded[i:i+filter.shape[0],j:j+filter.shape[1]]).sum()
			I_result[i][j] = I_result[i][j]/filter_sum

	#I_filtered_zeropadding = np.zeros((I.shape[0]*2+ 2*k0, I.shape[1]*2 + 2*k1))
	#for i in range(k0, k0+I.shape[0]*2):
	#	for j in range(k1, k1+I.shape[1]*2):
	#		for k in range(-k0, k0+1):
	#			for l in range(-k1, k1+1):
	#				I_filtered_zeropadding[i][j] = I_filtered_zeropadding[i][j] + filter[k0+k][k1+l]*I_zeropadded[i-2*k][j-2*l]
	#	I_filtered_zeropadding[i][j] = I_filtered_zeropadding[i][j]/filter_sum
	#Result = I_filtered_zeropadding[k0:k0+I.shape[0]*2,k1:k1+I.shape[1]*2]
	return I_result

def pyramid(I, levels):
	G = I.copy()
	gp = [G]
	for i in range(levels):
		#G = cv2.pyrDown(G) 
		G = DownSampling(G, kernel)
		#print("downsample level {}, has shape {}".format(i,G.shape))
		msg = "level{}.jpg".format(i)
		cv2.imwrite(msg,G)
		gp.append(G)
	return gp

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
kernel = getKernel(1,2)
B = filter.myImageFilter(I, kernel)

#B = gaussian_filter(I, sigma=1)
cv2.imwrite('Lenna_smoothed.jpg', B)

''' 2. Build a pyramid '''
levels = 6
P = pyramid(B, levels)
print("Build Gaussian Pyramid of size", levels)

''' 3. Subtract an interpolated coarser-level pyramid image from the original resolution blurred image'''
interpolated_level = 4
print("Use level {} image for interpolation and subtraction".format(interpolated_level))
I_interpolated = P[interpolated_level].copy()
for i in range(interpolated_level):
	I_interpolated = cv2.pyrUp(I_interpolated) #TODO
	#I_interpolated = UpSampling(I_interpolated)
	msg = "intepolation_level{}.jpg".format(i)
	cv2.imwrite(msg,I_interpolated)

print("shape after interpolation",I_interpolated.shape)
S = B - I_interpolated
cv2.imwrite("Lenna_diff.jpg",S)

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
print("Detected edge number: ", len(Edges))
Result = np.zeros(I.shape)
for i in Edges:
	x = int(i.x)
	y = int(i.y)
	Result[x][y] = 255
	#endpoints = i.endpoints
	#for p in endpoints:
	#	Result[p[0]][p[1]] = 255
cv2.imwrite("result.jpg",Result)






