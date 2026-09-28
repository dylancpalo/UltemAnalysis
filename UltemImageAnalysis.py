import sys
import cv2
import numpy as np
import time
import glob
TopBall = int(sys.argv[9])

sigma_squared = 0.0000001*0.0000001
pixToum = 2.2
umTomm = 1./1000.
milTomm = 0.0254
inTomm = 25.4
ToolingBallRadiusCalib = 6.348095
ToolingBallRadiusProd  = 6.351143

HalfX = 2592/2
HalfY = 1944/2
fStation=58
ToolingBallRadius = 0.75*0.5*25.4

def transformPixelX(xpix):
    xpix = (HalfX - xpix)
    xpix = xpix*pixToum
    xpix = xpix*umTomm
    return xpix

def transformPixelY(ypix):
    ypix = HalfY -ypix
    ypix = ypix*pixToum
    ypix = ypix*umTomm
    return ypix

def transformPixelR(rpix):
    rpix = rpix*pixToum
    rpix = rpix*umTomm
    rpix = rpix
    return rpix

def TransformToCameraCoordRadiusStationFids(xCCD, yCCD, rCCD):
    xp = transformPixelX(xCCD)
    yp = transformPixelY(yCCD)
    rp = transformPixelR(rCCD)

    zCAM = (ToolingBallRadius*fStation/rp) + fStation
    yCAM = yp*(zCAM-fStation)/fStation
    xCAM = xp*(zCAM-fStation)/fStation

    return xCAM, yCAM, zCAM


Directory = sys.argv[3]
sortedNames = sorted(glob.glob("%s*"%(Directory)))
print(sortedNames)
import numpy as np
import cv2 as cv
import sys
import time
from ImageAnalysis import *
from scipy.optimize import curve_fit 
import os
import glob
MinBkgLight = 30
OptimalBkgMedian=80
Width = 2592
Height = 1944
MaxInt = 255
BkgHalfWidth = 1000
printTime=1
#load image
makePlots=1
if (makePlots):
    from matplotlib import pyplot
    import matplotlib
initialTime = time.time()
if (printTime):
    t = time.time()


ImageDirectory=sys.argv[3]
stationID = int(sys.argv[7])
iteration = int(sys.argv[8])
sortedNames = sorted(glob.glob("%s*"%(ImageDirectory)))

for iImg in range(len(sortedNames)):
    print(len(sortedNames[iImg]), len(sys.argv[3]))
    ReducedName = sortedNames[iImg][len(sys.argv[3]):]
    print("Analyzing %s"%(ReducedName))

    TripletID = 1
    CAMID = 1#int(ReducedName[3:11])
    NImage = iImg#int(ReducedName[-8:-6])

    if (makePlots):
        matplotlib.rcParams['figure.figsize'] = (10, 3)
        fig, ax = pyplot.subplots(nrows=3,figsize=(6,8))
    im = cv.imread(sortedNames[iImg])
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(imgray, cv.HOUGH_GRADIENT, 0.1, 2000,
                               param1=60, param2=35,
                               minRadius=870, maxRadius=920)
    
    imgrayCirc = imgray
    if circles is not None:
        circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(imgrayCirc, center, 1, (255, 255, 255), 3)
            # circle outline
            radius = i[2]
            cv.circle(imgrayCirc, center, radius, (255, 0, 255), 1)
            if (makePlots):
                cv.imwrite("ImageOutputs/HoughCirc/HoughCircle.jpg", imgrayCirc) 
   
    
    circles = circles[0,:,:]
    centers = circles
    centers=centers[centers[:, 0].argsort()]
    centers = centers[0]
    
    Avg1X = centers[0]
    Avg1Y = centers[1]
    MinX1 = int(Avg1X - BkgHalfWidth)
    MaxX1 = int(Avg1X + BkgHalfWidth)
    MinY1 = int(Avg1Y - BkgHalfWidth)
    MaxY1 = int(Avg1Y + BkgHalfWidth)
    
    if (MinX1 < 0):
        MinX1 = int(0)
    if (MaxX1 > imgray.shape[1]-1):
        MaxX1 = int(imgray.shape[1]-1)

    if (MinY1 < 0):
        MinY1 = int(0)
    if (MaxY1 > imgray.shape[0]-1):
        MaxY1 = int(imgray.shape[0]-1)

            
    Bkg1 = imgray[MinY1:MaxY1,MinX1:MaxX1]
    IndexArrY = np.linspace(0, imgray.shape[0]-1, imgray.shape[0])
    IndexArrX = np.linspace(0, imgray.shape[1]-1, imgray.shape[1])
    IndexMeshX, IndexMeshY = np.meshgrid(IndexArrX, IndexArrY)
    
    IndexMeshX1 = IndexMeshX[MinY1:MaxY1,MinX1:MaxX1]
    IndexMeshY1 = IndexMeshY[MinY1:MaxY1,MinX1:MaxX1]
    
    IndexMeshX1Cut = IndexMeshX1[np.logical_not(Bkg1 <MinBkgLight)]
    IndexMeshY1Cut = IndexMeshY1[np.logical_not(Bkg1 <MinBkgLight)]
    Bkg1Cut = Bkg1              [np.logical_not(Bkg1 <MinBkgLight)]
    Bkg1CutMean = np.median(Bkg1Cut)
    def func(xy, a, b, c, d, e, f, g, h, i, j): 
        x, y = xy 
        return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x*x*x + h*y*y*y + i*x*x*y + j*y*y*x

    popt, pcov = curve_fit(func, (IndexMeshX1Cut, IndexMeshY1Cut), Bkg1Cut) 
    MeshZ1 = func((IndexMeshX1.flatten(), IndexMeshY1.flatten()), *popt) 
    MeshZ1=MeshZ1.reshape((int(len(MeshZ1)/(MaxX1-MinX1)), int(MaxX1-MinX1))) 
    
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    imgray[MinY1:MaxY1,MinX1:MaxX1] = \
        imgray[MinY1:MaxY1,MinX1:MaxX1]*(OptimalBkgMedian/MeshZ1)
    imgray = imgray.astype(np.uint8)

    
    '''
    def heatmap2d(i, j, arr: np.ndarray):
        im= ax[j].imshow(arr, cmap='viridis', vmin=0, vmax=200)
        colorbar = fig.colorbar(im)
    heatmap2d(0, 0, Bkg1)
    heatmap2d(0, 1, MeshZ1)
    heatmap2d(0, 2, imgray[MinY1:MaxY1,MinX1:MaxX1])
    pyplot.show()
    '''
    cv.imwrite("ImageOutputs/LightCorrected.bmp", imgray)
    

    #removing pixels associated with bad contours e.g. edges or stem of tooling ball 
    def FilterPixelsLightHoughBig(centers, pts, makePlots):
        if (makePlots):
            zerograyTemplate= np.zeros((Height, Width))
        indicesFilter = []
        for i in range(len(pts)):
            r = np.sqrt((centers[1]-pts[i,0])**2 + (centers[0]-pts[i,1])**2)
            if (abs(r-centers[2]) < 60):
                if (centers[1]-pts[i,1]< centers[0]-200):
                    indicesFilter.append(np.array([pts[i,0],pts[i,1]]))

        indicesFilter  = np.array(indicesFilter)
        if (makePlots):
            for i in range(len(indicesFilter)):
                zerograyTemplate[indicesFilter[i,0],indicesFilter[i,1]]=maxPixelIntensity
            cv.imwrite("ImageOutputs/ToolingBallHBigBall%i.jpg"%(0), zerograyTemplate)
        return indicesFilter



    zero = np.zeros(imgray.shape)
    ret, thresh = cv.threshold(imgray, thresholds, 255, 0) #make threshold image
    cv.imwrite("ImageOutputs/Threshold.jpg", thresh) 
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #grab contours
    c_select = []
    for c in contours:
        area = cv.contourArea(c)
        if (area < 5000):
            continue
        c_select.append(c)
    imgrayFresh = imgray
    zero = np.zeros(im.shape) 
    cv.drawContours(zero, c_select, -1, (0,255,0), 1)
    cv.imwrite("ImageOutputs/AllContoursOnBlack.jpg", zero) 
    centersHough = centers
    centers = []
    finalXIndices = []
    finalYIndices = []
    pts = np.where(zero == maxPixelIntensity)
    pts = np.vstack((pts[0], pts[1])).T
    ToolingBalls = FilterPixelsLightHoughBig(centersHough, pts, makePlots)  #remove bad pixels e.g. stem, edge effects
    if (printTime):
        elapsed = time.time() - t
        print("Process Contour Time: %f"%(elapsed))
        t = time.time()
    #loop over arrays of pixels
    #loop over the three balls 
    thisindices = np.array(ToolingBalls)
    #dont bother if there are no pixels
    if (len(thisindices)<10):
        print("BAD FIT 1")
    x = thisindices[:,0]
    y = thisindices[:,1]
    #fit (chi2 again)
    xc_2, yc_2, R_2, xprime, yprime = Fitter1(thisindices, [centersHough[1], centersHough[0], centersHough[2]], x, y, 0)
    #clean up based on residuals, tighter cut
    thisindices= RemoveOutliers(thisindices, ResThreshold2, x-xprime, y-yprime, 0)
    x = thisindices[:,0]
    y = thisindices[:,1]
    if (len(thisindices)<10):
        print("BAD FIT 1")
    #final fit chi2
    xc_2, yc_2, R_2, xprime, yprime = Fitter1(thisindices, [xc_2, yc_2, R_2], x, y, 1)
    
    #if there is a reasonable number of pixels, and it isn't the same contour, and the ratio of pixels to the radius is reasonable, accept!
    #print("%.2f, %.2f, %.2f, %.2f, %.2f"%(xc_2, yc_2, R_2, len(x), len(x)/R_2))
    if (len(x) > NRequiredPixels and len(x)/R_2 > PixelsToRadiusSelection):
        finalXIndices.append(x)
        finalYIndices.append(y)
        centers.append(np.array([yc_2, xc_2, R_2, len(x)]))
        if (makePlots):
            Ri = np.zeros(len(xprime))
            for i in range(len(xprime)):
                Ri[i] = np.sqrt((x[i]-xc_2)**2 + (y[i]-yc_2)**2)
            plot = ax[0].scatter(y,x,linewidths=0, alpha=.5,
                         edgecolor='k',
                         s = 100,
                         c=Ri-R_2)
            ax[1].hist(Ri-R_2, np.linspace(-2, 2, 40), label="sig=%.2f [pix]"%(np.sqrt(np.var(Ri-R_2))))
            ax[1].set_ylim([0, 500])
            ax[1].set_xlim([-1.5, 1.5])
            ax[1].legend()
    centers = np.array(centers)[0,:]
    print("Found Tooling Balls")
    print("X [pix] ,  Y [pix],  R [pix],  NPix")
    print("%8.2f, %8.2f, %8.2f, %8.2f"%(centers[0], centers[1], centers[2], centers[3]))     
    #-------------------------------------------------------------------
    xCAM, yCAM, zCAM = TransformToCameraCoordRadiusStationFids(centers[0], centers[1], centers[2])

    csvrow = "%s, %i, %i, %i, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f,  %.4f\n"%(ReducedName, TripletID, CAMID, NImage,
        centers[0], centers[1], centers[2], Bkg1CutMean,\
        xCAM, yCAM, zCAM)
    with open('%sBigBallCCD.csv'%(Directory),'a') as fd:
            fd.write(csvrow)

    if (makePlots):
        #residual plots for the three tooling balls are now saved
        fig.colorbar(plot, ax=ax.ravel().tolist())
        pyplot.savefig('ImageOutputs/ResidualPlots.jpg', bbox_inches='tight')
           
    if (makePlots):
        #plot the final contours on a black image
        for j in range(len(finalXIndices)):
            imgray[finalXIndices[j],finalYIndices[j]]=maxPixelIntensity
        cv.imwrite("ImageOutputs/finalContours.jpg", imgrayFresh)
TotalCamArray = np.genfromtxt("%sBigBallCCD.csv"%(Directory), delimiter=", ")
