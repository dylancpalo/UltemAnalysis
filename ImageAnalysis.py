import numpy as np
import sys
import cv2 as cv
#from scipy      import optimize
from scipy.optimize import minimize

HeightOfBottom = 80
rExpected = 110
Width = 2592
Height = 1944
RMin = 0
NToolingBalls = 3
NRequiredPixels= 350
RequiredDistanceAwayFromFoundContour = 100 #dont want any repeats
YBallOffset = 80 #get the y of the ball just using the max - this
YBallOffsetLight = 350
YBallOffsetBig = 800


YEdgeWidth1 = 100# kill all pixels this close to the top
YEdgeWidth2 = 100# kill all the pixels this close to the bottom
XEdgeWidth1 = 3#40  # kill all the pixels this close to the sides
XEdgeWidth2 = 3#40  # kill all the pixels this close to the sides
ContourLimiter= 100#-10#100#-10
ContourLimiterLight = 120
ContourLimiterLight2 = 100
RMaxLight = YBallOffsetLight -80
RMax = 230


RMaxBig = 800
StemCut = 10
PixelsToRadiusSelection = 2.3# basically regardless of R, want the number of pixels in the fit /R greater than this
XEdge1 = 2592/3#cut the grid into the 1st-2nd tooling ball
XEdge2 = 2*2592/3  #cut the grid into the 2nd-3rd tooling ball
sigma_squared = 0.0000001*0.0000001#tooling ball
maxPixelIntensity = 255
ResThreshold1 =3
ResThreshold2 =float(sys.argv[5])
thresholds = int(sys.argv[6])


#simple radius calculation for a given point
def calc_R(xc, yc, x, y):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

#minimization function, minimize the R given an x,y (c)
def f_2(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c, x, y)
    return Ri - Ri.mean()

#calculate a chi2  
def CircleChi2(par, x, y):
	ang =  np.arctan2(y - par[1], x - par[0])
	xp = par[2]*np.cos(ang)
	yp = par[2]*np.sin(ang) 
	dx = (x - par[0]) - xp
	dy = (y - par[1]) - yp

	return np.sum(abs(dx) + abs(dy))#/sigma_squared
#calculate the prime x,y i.e. given xc,yc, r what is xp and yp
def CirclePrime(par, x, y):
	xn = x - par[0]
	yn = y - par[1]
	ang =  np.arctan2(y - par[1], x - par[0])
	xp = par[2]*np.cos(ang) + par[0]
	yp = par[2]*np.sin(ang) + par[1]
	return xp, yp

#chi2 fit implementation
def CircleFit(x, y, p0, finalFit):
	maxIterations=500
	xTol = 0.5
	faTol = 1e-1
	if (finalFit):
		maxIterations = 1000
		xTol = 0.0005
		faTol  = 1e-3

	residueinfo2 = minimize(CircleChi2, p0, args=(x, y), method='Nelder-Mead',  options={'maxiter': maxIterations, 'xatol':xTol, 'fatol':faTol})
	#print(residueinfo2.fun)

	if (residueinfo2.success!=1 and finalFit==1):
		print(residueinfo2)
	xprime, yprime = CirclePrime(residueinfo2.x, x, y)
	#pyplot.hist(xprime - x, np.linspace(-10, 10, 100))
	#pyplot.hist(yprime - y, np.linspace(-10, 10, 100))
	#pyplot.show()
	return residueinfo2.x, xprime, yprime


#removing pixels associated with bad contours e.g. edges or stem of tooling ball 
def FilterPixels(zerogray, XNominal, YNominal, XOffset, YOffset, iball):
	zerograyTemplate= np.zeros((3, zerogray.shape[0], zerogray.shape[1]))
	minX = 9999
	maxX = -999
	minY = 9999
	maxY = -999
	minR = 9999
	indicesFilter = [ [], [], []]
	selectedIndices = [ [], [], []]

	for i in range(zerogray.shape[0]):
		for j in range(zerogray.shape[1]):
			if (zerogray[i,j]!=0 and i > YEdgeWidth1 and i < zerogray.shape[0] - YEdgeWidth2 and j > XEdgeWidth1 and j < zerogray.shape[1] - XEdgeWidth2):
				if (j < XEdge1):
					indicesFilter[0].append(np.array([i,j]))
				if (j > XEdge1 and j < XEdge2):
					indicesFilter[1].append(np.array([i,j]))
				if (j > XEdge2):
					indicesFilter[2].append(np.array([i,j]))

	for i in range(len(indicesFilter)):
		dynamicList = []
		minCont = np.min(indicesFilter[i], axis=0)
		maxCont = np.max(indicesFilter[i], axis=0)
		indicesFilter[i]  = np.array(indicesFilter[i])
		#print(minCont, maxCont)
		for j in range(len(indicesFilter[i])):
			#print(indicesFilter[i][j,0], indicesFilter[i][j,1])
			if (indicesFilter[i][j,0] > minCont[0]+ContourLimiter and \
				indicesFilter[i][j,0] < maxCont[0]-ContourLimiter):
				dynamicList.append(indicesFilter[i][j,:])

		indicesFilter[i] = np.array(dynamicList)
		#print("new shape: ", indicesFilter[i].shape)

	C = np.zeros((len(indicesFilter), 2))
	for i in range(len(indicesFilter)):
		maxindex = np.argmin(indicesFilter[i][:,0])
		C[i,0] = indicesFilter[i][maxindex,0]+YBallOffset
		C[i,1] = indicesFilter[i][maxindex,1]

	for index in range(len(indicesFilter)):
		for i in range(len(indicesFilter[index])):
			ix = indicesFilter[index][i,0]
			iy = indicesFilter[index][i,1]
			r = np.sqrt( (ix-C[index,0])**2 + (iy-C[index,1])**2)
			if (r < RMax and r>RMin):
				selectedIndices[index].append(indicesFilter[index][i,:])
				zerograyTemplate[index, indicesFilter[index][i,0],indicesFilter[index][i,1]]=maxPixelIntensity

	for index in range(len(indicesFilter)):
		cv.imwrite("ImageOutputs/ToolingBall%i.jpg"%(index), zerograyTemplate[index,:,:])
	return selectedIndices




#removing pixels associated with bad contours e.g. edges or stem of tooling ball 
def FilterPixelsLight(zerogray, XNominal, YNominal, XOffset, YOffset, iball):
	zerograyTemplate= np.zeros((3, zerogray.shape[0], zerogray.shape[1]))
	minX = 9999
	maxX = -999
	minY = 9999
	maxY = -999
	minR = 9999	

	indicesFilter = [ [], [], []]
	selectedIndices = [ [], [], []]

	for i in range(zerogray.shape[0]):
		for j in range(zerogray.shape[1]):
			if (zerogray[i,j]!=0 and i > YEdgeWidth1 and i < zerogray.shape[0] - YEdgeWidth2 and j > XEdgeWidth1 and j < zerogray.shape[1] - XEdgeWidth2):
				if (j < XEdge1):
					indicesFilter[0].append(np.array([i,j]))
				if (j > XEdge1 and j < XEdge2):
					indicesFilter[1].append(np.array([i,j]))
				if (j > XEdge2):
					indicesFilter[2].append(np.array([i,j]))

	#find the center of the LED and kill things around it
	C = np.zeros((len(indicesFilter), 2))
	LeftEdge = np.zeros((len(indicesFilter)))
	RightEdge = np.zeros((len(indicesFilter)))
	TopEdge = np.zeros((len(indicesFilter)))

	for i in range(len(indicesFilter)):
		indicesFilter[i]  = np.array(indicesFilter[i])
		maxindex = np.argmin(indicesFilter[i][:,0])
		C[i,0] = indicesFilter[i][maxindex,0]+YBallOffsetLight
		C[i,1] = indicesFilter[i][maxindex,1]# + 20	
		LeftEdge[i]  = np.min(indicesFilter[i][:,1])
		RightEdge[i] = np.max(indicesFilter[i][:,1])
		TopEdge[i] = np.min(indicesFilter[i][:,0])
		zerograyTemplate[i, int(C[i,0]),int(C[i,1])]=maxPixelIntensity
		
	for index in range(len(indicesFilter)):
		for i in range(len(indicesFilter[index])):
			i0 = indicesFilter[index][i,0]
			i1 = indicesFilter[index][i,1]
			r = np.sqrt( (i1-C[index,1])**2 + (i0-C[index,0])**2)
			if (r < RMaxLight and r>RMin):
				selectedIndices[index].append(indicesFilter[index][i,:])



	#now the thing at the top is the ball! now kill everything except the ball
	indicesFilter = selectedIndices
	selectedIndices = [ [], [], []]	
	C = np.zeros((len(indicesFilter), 2))
	for i in range(len(indicesFilter)):
		indicesFilter[i]  = np.array(indicesFilter[i])
		maxindex = np.argmin(indicesFilter[i][:,0])
		C[i,0] = indicesFilter[i][maxindex,0]+YBallOffset
		C[i,1] = indicesFilter[i][maxindex,1]
		zerograyTemplate[i, int(C[i,0]),int(C[i,1])]=maxPixelIntensity
	
	for index in range(len(indicesFilter)):
		for i in range(len(indicesFilter[index])):
			i0 = indicesFilter[index][i,0]
			i1 = indicesFilter[index][i,1]
			r = np.sqrt( (i1-C[index,1])**2 + (i0-C[index,0])**2)
			if (r < RMax and r>RMin):
				selectedIndices[index].append(indicesFilter[index][i,:])
				zerograyTemplate[index, indicesFilter[index][i,0],indicesFilter[index][i,1]]=maxPixelIntensity

	for index in range(len(indicesFilter)):
		cv.imwrite("ImageOutputs/ToolingBallT%i.jpg"%(index), zerograyTemplate[index,:,:])
	return selectedIndices




#removing pixels associated with bad contours e.g. edges or stem of tooling ball 
def FilterPixelsLightHough(centers, pts, makePlots):
	if (makePlots):
		zerograyTemplate= np.zeros((3, Height, Width))
	indicesFilter = [ [], [], []]
	ROffset = 5
	YOffset = 30
	TopBallCut = 8
	for i in range(len(pts)):
			if   (np.sqrt( (pts[i,1]-centers[0, 0])**2+ (pts[i,0]-centers[0, 1])**2) < centers[0,2] + ROffset and pts[i,0]-centers[0, 1]+ YOffset  < centers[0,2]+ ROffset \
				and pts[i,0]-(centers[0, 1]-centers[0,2])> TopBallCut):
				indicesFilter[0].append(np.array([pts[i,0],pts[i,1]]))
			elif (np.sqrt( (pts[i,1]-centers[1, 0])**2+ (pts[i,0]-centers[1, 1])**2) < centers[1,2] + ROffset and pts[i,0]-centers[1, 1]+ YOffset  < centers[1,2] + ROffset\
				and pts[i,0]-(centers[1, 1]-centers[1,2])> TopBallCut):
				indicesFilter[1].append(np.array([pts[i,0],pts[i,1]]))
			elif (np.sqrt( (pts[i,1]-centers[2, 0])**2+ (pts[i,0]-centers[2, 1])**2) < centers[2,2] + ROffset and pts[i,0]-centers[2, 1]+ YOffset  < centers[2,2]+ ROffset \
				and pts[i,0]-(centers[2, 1]-centers[2,2])> TopBallCut):

				indicesFilter[2].append(np.array([pts[i,0],pts[i,1]]))

	for index in range(len(indicesFilter)):
		indicesFilter[index]  = np.array(indicesFilter[index])
		if (makePlots):
			for i in range(len(indicesFilter[index])):
				zerograyTemplate[index, indicesFilter[index][i,0],indicesFilter[index][i,1]]=maxPixelIntensity
			cv.imwrite("ImageOutputs/ToolingBallH%i.jpg"%(index), zerograyTemplate[index,:,:])
	return indicesFilter



#removing pixels associated with bad contours e.g. edges or stem of tooling ball 
def FilterPixelsLightSingle(zerogray, XNominal, YNominal, XOffset, YOffset, iball):
	zerograyTemplate= np.zeros((3, zerogray.shape[0], zerogray.shape[1]))
	minX = 9999
	maxX = -999
	minY = 9999
	maxY = -999
	minR = 9999	
	RMin = 750
	RMaxLightBig = 950
	#YBallOffsetBigLight = 900
	StemCutBig =700

	indicesFilter = [ []]
	selectedIndices = [ []]
	XEdgeOffset = 50
	YEdgeOffset = 50
	for i in range(zerogray.shape[0]):
		for j in range(zerogray.shape[1]):
			if (zerogray[i,j]!=0 and i > YEdgeOffset and i < zerogray.shape[0] - YEdgeOffset and j > XEdgeOffset and j < zerogray.shape[1] - XEdgeOffset):
				indicesFilter[0].append(np.array([i,j]))
	#find the center of the LED and kill things around it

	C = np.zeros((len(indicesFilter), 2))
	for i in range(len(indicesFilter)):
		indicesFilter[i]  = np.array(indicesFilter[i])
		maxindex = np.argmin(indicesFilter[i][:,0])
		C[i,0] =(np.max(indicesFilter[i][:,0])+np.min(indicesFilter[i][:,0]))/2#indicesFilter[i][maxindex,0]+YBallOffsetBigLight
		C[i,1] =(np.max(indicesFilter[i][:,1])+np.min(indicesFilter[i][:,1]))/2#indicesFilter[i][maxindex,1]
		#print(C[i,:])
		zerograyTemplate[i, int(C[i,0]),int(C[i,1])]=maxPixelIntensity
	
	for index in range(len(indicesFilter)):
		for i in range(len(indicesFilter[index])):
			i0 = indicesFilter[index][i,0]
			i1 = indicesFilter[index][i,1]
			r = np.sqrt( (i1-C[index,1])**2 + (i0-C[index,0])**2)
			if (r < RMaxLightBig and i0-C[index,0] < StemCutBig):
				selectedIndices[index].append(indicesFilter[index][i,:])



	#now the thing at the top is the ball! now kill everything except the ball
	indicesFilter = selectedIndices
	selectedIndices = [ [], [], []]	
	C = np.zeros((len(indicesFilter), 2))
	for i in range(len(indicesFilter)):
		indicesFilter[i]  = np.array(indicesFilter[i])
		maxindex = np.argmin(indicesFilter[i][:,0])
		C[i,0] = indicesFilter[i][maxindex,0]+YBallOffsetBig
		C[i,1] = indicesFilter[i][maxindex,1]
		#zerograyTemplate[i, int(C[i,0]),int(C[i,1])]=maxPixelIntensity
	for index in range(len(indicesFilter)):
		for i in range(len(indicesFilter[index])):
			i0 = indicesFilter[index][i,0]
			i1 = indicesFilter[index][i,1]
			r = np.sqrt( (i1-C[index,1])**2 + (i0-C[index,0])**2)
			#if (r > RMin):
			selectedIndices[index].append(indicesFilter[index][i,:])
			zerograyTemplate[index, indicesFilter[index][i,0],indicesFilter[index][i,1]]=maxPixelIntensity

	for index in range(len(indicesFilter)):
		cv.imwrite("ImageOutputs/ToolingBallBig%i.jpg"%(index), zerograyTemplate[index,:,:])
	return selectedIndices



#kill outliers in array of pixels based on residuals from fit
def RemoveOutliers(indices, ResThreshold, dx, dy, final):
	killList = []
	maxdX = np.max(indices[:,0]) #actually negative Y to remove stem
	for i in range(len(dx)):
		if abs(dx[i]) > ResThreshold or abs(dy[i]) > ResThreshold:
			killList.append(i)
		if (maxdX - indices[i,0] <StemCut and final==1):
			killList.append(i)
	x = indices[:,0]
	y = indices[:,1]
	x = np.delete(x, killList) 
	y = np.delete(y, killList) 
	indices = np.vstack((x, y)).T
	return indices


#chi2 fit
def Fitter1(indices, p0, x, y, finalFit):
	'''
	if (p0[0]==0):
		x_m = np.mean(x)
		y_m = np.mean(y)
		center_estimate = x_m, y_m
		center_2, ier = optimize.leastsq(f_2, center_estimate, args=(x, y), ftol=1e-3)
		xc_2, yc_2 = center_2
		Ri_2       = calc_R(*center_2, x, y)
		R_2        = Ri_2.mean()
		p0 = [xc_2, yc_2, R_2]
	'''
	Par, xprime, yprime = CircleFit(x, y, p0, finalFit)
	return Par[0], Par[1], Par[2], xprime, yprime#xc_2, yc_2, R_2, Ri_2, residu_2

#least sq fit
def Fitter2(indices, p0, x, y):
	x_m = np.mean(x)
	y_m = np.mean(y)
	center_estimate = x_m, y_m
	center_2, ier = optimize.leastsq(f_2, center_estimate, args=(x, y))
	xc_2, yc_2 = center_2
	Ri_2       = calc_R(*center_2, x, y)
	R_2        = Ri_2.mean()
	p0 = [xc_2, yc_2, R_2]
	ang =  np.arctan2(y - p0[1], x - p0[0])
	xp = p0[2]*np.cos(ang) + p0[0]
	yp = p0[2]*np.sin(ang) + p0[1]
	#for i in range(len(x)):
	#	print("%.2f, %.2f, %.2f, %.2f"%(x[i], xp[i], y[i], yp[i]))
	return p0[0], p0[1], p0[2], xp, yp
