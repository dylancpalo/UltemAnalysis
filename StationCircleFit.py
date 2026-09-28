import numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import cm
import matplotlib as mpl
import math
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib.cm as c
import glob
import csv

nPanelsPerStation=12

R= 822
FrameOrdering=[8,5,9,16,12,11,14,7,6,17,3,4,1,0,10,15,2,13]
FolderString = "Station_NPYFiles06_25_FullCalibMod"
def sort_and_label(arr):
    """
    arr: (36,3) array with columns [X, Y, Z]

    Returns:
        array with columns [X, Y, Z, label, phi]
        sorted by (label, phi)
    """

    # --- Step 1: sort by Z (3rd column) ---
    arr = arr[np.argsort(arr[:, 2])]
    

    # --- Step 2: assign labels (0,1,2,3) in blocks of 9 ---
    labels = np.repeat(np.arange(4), 9)

    # --- Step 3: compute phi = atan2(Y, X) ---
    phi = np.arctan2(arr[:, 1], arr[:, 0])

    # --- Step 4: combine everything ---
    out = np.column_stack([arr, labels, phi])

    # --- Step 5: sort by label, then phi ---
    idx = np.lexsort((out[:, 4], out[:, 3]))  # (phi, label)
    out = out[idx]
    print("")
    for i in range(len(out)):
    	print("%8.3f, %8.3f, %8.3f, %8.3f, %8.3f"%(out[i,0], out[i,1], out[i,2], out[i,3], out[i,4]))
    return out[:,:3]



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

def func(xy, a, b, c, d): 
    x, y = xy 
    return a + b*x/R + c*y/R + d*(y**2)/(R*R) 


def SplitIntoXYZ(array):
	return array[:,:,0].flatten(), array[:,:,1].flatten(), array[:,:,2].flatten()


def FitCurve(X, Y, Z):
	poptArray = np.zeros((4, 4))
	meanResArray =np.zeros(4)
	stdResArray = np.zeros(4)
	for i in range(4):
		popt, pcov = curve_fit(func, (X[i*9:(i+1)*9], Y[i*9:(i+1)*9]), Z[i*9:(i+1)*9]) 
		ZP = func((X[i*9:(i+1)*9], Y[i*9:(i+1)*9]), *popt) 
		poptArray[i,:]= popt.flatten()
		meanResArray[i] =np.mean(Z[i*9:(i+1)*9]-ZP)
		stdResArray[i]  =np.sqrt(np.var(Z[i*9:(i+1)*9]-ZP))
	return poptArray, meanResArray, stdResArray


def StoreAndSliceData(Station_Index):
    FidsStation = np.load("%s/FidsStation%i_mech1.npy"%(FolderString, Station_Index))
    FidsTracker = np.load("%s/FidsTracker%i_mech1.npy"%(FolderString, Station_Index))
    PanelIDs = np.load("%s/StationPanelIDs%i_mech1..npy"%(FolderString, Station_Index))
    print(FidsTracker.shape)
    panel888Ordering = np.genfromtxt("panel_output.txt", delimiter=", ")
    print(panel888Ordering)
    CSVFileName = "TrackerFids_IDs.csv"
    CSVFileNameSorted = "TrackerFids_IDsSorted.csv"

    
    for i in range(len(PanelIDs)):
        for j in range(len(panel888Ordering)):
            if (PanelIDs[i]==panel888Ordering[j,0]):
                CSVFile = "%i, %i, %i, %i, %i, %i, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f\n"%(\
                panel888Ordering[j,3], panel888Ordering[i,4], FrameOrdering[Station_Index], Station_Index, PanelIDs[i], i, \
                FidsTracker[i,0,0], FidsTracker[i,0,1], FidsTracker[i,0,2], \
                FidsTracker[i,1,0], FidsTracker[i,1,1], FidsTracker[i,1,2], \
                FidsTracker[i,2,0], FidsTracker[i,2,1], FidsTracker[i,2,2])
                print(CSVFile)
                with open(CSVFileName, "a") as file:
                    file.write(CSVFile)
                break
                
    
    with open(CSVFileName, newline="") as f:
        reader = csv.reader(f)
        # sort by column 0, then column 1
    
    #sorted_rows = sorted(reader, key=lambda row: (float(row[0]), float(row[1])))
    #with open(CSVFileNameSorted,w, newline="") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(sorted_rows)
    
    XStation = FidsStation[:,:,0].flatten()
    YStation = FidsStation[:,:,1].flatten()
    ZStation = FidsStation[:,:,2].flatten()
    XStation0 = XStation[np.logical_not(ZStation>-50)]
    YStation0 = YStation[np.logical_not(ZStation>-50)]
    ZStation0 = ZStation[np.logical_not(ZStation>-50)]
    
    XStation1 = XStation[np.logical_not(ZStation<-50)]
    YStation1 = YStation[np.logical_not(ZStation<-50)]
    ZStation1 = ZStation[np.logical_not(ZStation<-50)]

    
    
    circOptS0, XP, YP = CircleFit(XStation0, YStation0, [0.1, 0.1, R], 1)
    circOptS1, XP, YP = CircleFit(XStation1, YStation1, [0.1, 0.1, R], 1)
    circOptS, XP, YP = CircleFit(XStation, YStation, [0.1, 0.1, R], 1)
    
    print("%i, %i, %6.3f, %6.3f, %6.3f"%(2, Station_Index,  circOptS[0], circOptS[1], circOptS[2]))
    print("%i, %i, %6.3f, %6.3f, %6.3f"%(0, Station_Index, circOptS0[0], circOptS0[1], circOptS0[2]))
    print("%i, %i, %6.3f, %6.3f, %6.3f"%(1, Station_Index, circOptS1[0], circOptS1[1], circOptS1[2]))
    print("")
    #circOptS = circOptS0
    
    circOptT, XP, YP = CircleFit(FidsTracker[:,:,0].flatten(), FidsTracker[:,:,1].flatten(), [0.1, 0.1, R], 1)
    poptS, meanResS,sigResS = FitCurve(FidsStation[:,:,0].flatten(), FidsStation[:,:,1].flatten(), FidsStation[:,:,2].flatten())
    poptT, meanResT,sigResT = FitCurve(FidsTracker[:,:,0].flatten(), FidsTracker[:,:,1].flatten(), FidsTracker[:,:,2].flatten())
    
    WiresStation0 = 0#np.load("%s/WiresTracker1%i_mech1.npy"%(FolderString, Station_Index))
    WiresStation1 = 0#np.load("%s/WiresTracker0%i_mech1.npy"%(FolderString, Station_Index))
    
    return FidsStation, circOptS, poptS, FidsTracker, circOptT, poptT, PanelIDs, WiresStation0, WiresStation1

def PrintCircleFits(Station_Index, Station_Tracker, circOpt, popt):
	#print("------------------------------------------------------------------------------------------------")
	#print(Station_Tracker, Station_Index)
	for i in range(4):
		print("%i,%10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f"%(i,circOpt[0], circOpt[1],circOpt[2], popt[i,0], popt[i,1], popt[i,2], popt[i,3]))
	##print("------------------------------------------------------------------------------------------------")
	#print("")
	#print("------------------------------------------------------------------------------------------------")


fig, ax = pyplot.subplots(nrows=2, ncols=4,figsize=(20,8))
ax[0,0].set_xlabel("X Metrology Stand [mm]")
ax[0,0].set_ylabel("Y Metrology Stand [mm]")
ax[0,1].set_xlabel("X Metrology Stand [mm]")
ax[0,1].set_ylabel("Z Metrology Stand [mm]")
ax[0,2].set_xlabel("Z Metrology Stand [mm]")
ax[0,2].set_ylabel("Y Metrology Stand [mm]")
ax[0,3].set_xlabel("Ce X Metrology Stand [mm]")
ax[0,3].set_ylabel("Ce Y Metrology Stand [mm]")
ax[1,0].set_xlabel("X Frame [mm]")
ax[1,0].set_ylabel("Y Frame [mm]")
ax[1,1].set_xlabel("X Frame [mm]")
ax[1,1].set_ylabel("Z Frame [mm]")
ax[1,2].set_xlabel("Z Frame [mm]")
ax[1,2].set_ylabel("Y Frame [mm]")
ax[1,3].set_xlabel("Ce X Frame [mm]")
ax[1,3].set_ylabel("Ce Y Frame [mm]")

StationFiles = glob.glob("%s/FidsStation*npy"%(FolderString))#[0][-5]
StationIDs = []
for i in range(18):
	StationIDs.append(i)
	'''
	if (len(StationFiles[i])==33): #single digit
		StationIDs.append(int(StationFiles[i][-5]))
	else:
		StationIDs.append(int(StationFiles[i][-6:-4]))
	'''
StationIDs = np.sort(np.array(StationIDs))

color = cm.rainbow(np.linspace(0, 1, 12))

#fig = pyplot.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim(-800, 800) 
#ax.set_ylim(-800, 800)

circOptSMaster = []
for i in range(len(StationFiles)):

	#fig = pyplot.figure()
	#ax = pyplot.axes()#projection ='3d')


	FidsStation, circOptS, poptS, FidsTracker, circOptT, poptT, panelIDs, WiresStation0, WiresStation1 = StoreAndSliceData(StationIDs[i])
	circOptSMaster.append(circOptS)

	'''
	for j in range(int(nPanelsPerStation)):
		for k in range(WiresStation0.shape[1]):
			xx = np.vstack([WiresStation0[j,k,0],WiresStation1[j,k,0]])
			yy = np.vstack([WiresStation0[j,k,1],WiresStation1[j,k,1]])
			zz = np.vstack([WiresStation0[j,k,2],WiresStation1[j,k,2]])
			ax.plot(xx, yy, zz, color=color[j], marker='o',markersize=0.1)


		#ax.plot(xx,yy, color=color[j],label="Panel %i, MN%i, Z: %5.1f"%(j, panelIDs[j], np.median(WiresStation0[j,:,2])), linewidth=0.3)
		#handles, labels = pyplot.gca().get_legend_handles_labels()
		#by_label = dict(zip(labels, handles))
		#ax.legend(by_label.values(), by_label.keys())
		#ax.set_title("Station %i"%(StationIDs[i]))
	#pyplot.show()
	'''
	#PrintCircleFits(StationIDs[i], "Station", circOptS, poptS)
	#PrintCircleFits(StationIDs[i], "Tracker", circOptT, poptT)
        
	marker='o'
	if (FrameOrdering[i]%2==0):
		marker='x'
        
	ax[0,0].scatter(FidsStation[:,:,0].flatten(), FidsStation[:,:,1].flatten(), marker=marker, label="%2i: Xc=%4.2f Yc=%4.2f Rc=%4.2f"%(StationIDs[i], circOptS[0], circOptS[1], circOptS[2]))
	ax[0,1].scatter(FidsStation[:,:,0].flatten(), FidsStation[:,:,2].flatten(), marker=marker, label="%i"%(StationIDs[i]))
	ax[0,2].scatter(FidsStation[:,:,2].flatten(), FidsStation[:,:,1].flatten(), marker=marker, label="%i"%(StationIDs[i]))
	ax[0,3].scatter(circOptS[0], circOptS[1], marker=marker, label="%2i, %1i"%(StationIDs[i], FrameOrdering[i]%2))

	ax[1,0].scatter(FidsTracker[:,:,0].flatten(), FidsTracker[:,:,1].flatten(), marker=marker, label="%2i: Xc=%4.2f Yc=%4.2f Rc=%4.2f"%(StationIDs[i], circOptT[0], circOptT[1], circOptT[2]))
	ax[1,1].scatter(FidsTracker[:,:,0].flatten(), FidsTracker[:,:,2].flatten(), marker=marker, label="%i"%(StationIDs[i]))
	ax[1,2].scatter(FidsTracker[:,:,2].flatten(), FidsTracker[:,:,1].flatten(), marker=marker, label="%i"%(StationIDs[i]))
	ax[1,3].scatter(circOptT[0], circOptT[1], marker=marker, label="%2i, %1i"%(StationIDs[i], FrameOrdering[i]%2))


	NewCSV = np.vstack((FidsTracker[:,:,0].flatten(), FidsTracker[:,:,1].flatten(), FidsTracker[:,:,2].flatten())).T
	NewCSV = sort_and_label(NewCSV)
	np.savetxt('real_csvs/station%02i.csv'%(StationIDs[i]), NewCSV, delimiter=',',fmt='%8.3f')
    


	


ax[0,0].legend()
ax[0,1].legend()
ax[0,2].legend()
ax[0,3].legend()

ax[1,0].legend()
ax[1,1].legend()
ax[1,2].legend()
ax[1,3].legend()

pyplot.show()

pyplot.savefig("Stations.pdf")



'''
circOptSMaster = np.array(circOptSMaster)
fig, ax = pyplot.subplots(nrows=1, ncols=3,figsize=(20,8))
ax[0].hist(circOptSMaster[:,0], bins=np.linspace(-2, 2, 100))
ax[1].hist(circOptSMaster[:,1], bins=np.linspace(-2, 2, 100))
ax[2].hist(circOptSMaster[:,2], bins=np.linspace(822, 822.5, 100))
pyplot.show()
'''
