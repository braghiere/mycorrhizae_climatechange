from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap, cm, shiftgrid
from netCDF4 import Dataset, date2index
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import numpy.ma as ma
import numpy as np
from mpl_toolkits.basemap import maskoceans, interp
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
import scipy.stats as stats

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)/(window_size))
    return np.convolve(interval,window,'same')

def smooth1(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_padded = np.pad(y, (box_pts//2,box_pts-1-box_pts//2), mode='edge')
    y_smooth = np.convolve(y_padded, box, mode='valid')
    return y_smooth

# High resolution cost lines
#http://introtopython.org/visualization_earthquakes.html
# High resolution cost lines
#http://basemaptutorial.readthedocs.io/en/latest/utilities.html
plt.style.use('ggplot')
SIZE = 48
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)  # # size of the figure title


f = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50Bgc/IHistClm50Bgc.NPP_NUPTAKE.nc','r')
fi = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50Bgc/IHistClm50Bgc.NUPTAKE.nc','r')
fj = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50Bgc/IHistClm50Bgc.PLANT_NDEMAND.nc','r')
fk = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50Bgc/IHistClm50Bgc.NPP.nc','r')
f1 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSulman/IHistClm50BgcSulman.NPP_NUPTAKE.nc','r')
f1i = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSulman/IHistClm50BgcSulman.NUPTAKE.nc','r')
f1j = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSulman/IHistClm50BgcSulman.PLANT_NDEMAND.nc','r')
f1k = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSulman/IHistClm50BgcSulman.NPP.nc','r')
f2 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.NPP_NUPTAKE.nc','r')
f2i = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.NUPTAKE.nc','r')
f2j = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.PLANT_NDEMAND.nc','r')
f2k = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.NPP.nc','r')
f3 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.NPP_NUPTAKE.nc','r')
f3i = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.NUPTAKE.nc','r')
f3j = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.PLANT_NDEMAND.nc','r')
f3k = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.NPP.nc','r')
#future map
f4 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/ISSP585Clm50BgcCrop_all/ISSP585Clm50BgcCrop_all.NPP_NUPTAKE.nc','r')
f4i = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/ISSP585Clm50BgcCrop_all/ISSP585Clm50BgcCrop_all.NUPTAKE.nc','r')
f4j = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/ISSP585Clm50BgcCrop_all/ISSP585Clm50BgcCrop_all.PLANT_NDEMAND.nc','r')
f4k = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/ISSP585Clm50BgcCrop_all/ISSP585Clm50BgcCrop_all.NPP.nc','r')

surf = Dataset('/home/renato/Steindinger/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190304.nc','r')
params = Dataset('/home/renato/Steindinger/clm5_params.c171117.nc','r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time'][:]
time_future = f4.variables['time'][:]
print time
var = 'NPP_NUPTAKE'
nee = f1.variables[var]
unit = nee.units
long_name = nee.long_name
nee_stein1 = f1.variables[var][:]
nee_stein2 = f2.variables[var][:]
nee_stein3 = f3.variables[var][:]
nee_stein4 = f4.variables[var][:]
var = 'NPP_NUPTAKE'
nee = f.variables[var][:]
nee_stein1 = f1.variables[var][:]
var = 'NUPTAKE'
neei = fi.variables[var][:]
neei_stein1 = f1i.variables[var][:]
neei_stein2 = f2i.variables[var][:]
neei_stein3 = f3i.variables[var][:]
neei_stein4 = f4i.variables[var][:]
var = 'PLANT_NDEMAND'
neej = fj.variables[var][:]
neej_stein1 = f1j.variables[var][:]
neej_stein2 = f2j.variables[var][:]
neej_stein3 = f3j.variables[var][:]
neej_stein4 = f4j.variables[var][:]
var = 'NPP'
neek = fk.variables[var][:]
neek_stein1 = f1k.variables[var][:]
neek_stein2 = f2k.variables[var][:]
neek_stein3 = f3k.variables[var][:]
neek_stein4 = f4k.variables[var][:]

# Function to calculate the weights of latitude
#1.875 longitude X 1.25 latitude)
radius = 6367449
res_lon = 1.875
#res_lon = 1.90
res_lat = 1.25
m_pi = 3.14159265358979323846
dtor = 360./(2*m_pi)

lon_u = res_lon/dtor
lon_l = 0.0/dtor

#True weight
lat_l = lat/dtor
lat_u = (lat + res_lat)/dtor

weights = (radius*radius*(lon_u - lon_l)*(np.sin(lat_u)-np.sin(lat_l)))

#Tropical weight
lat_trop = np.linspace(lat[0],lat[-1],num=len(lat),endpoint=True)
lat_trop[abs(lat_trop) > 23.5] = 90.0

lat_l = lat_trop/dtor
lat_u = (lat_trop + res_lat)/dtor

weights_trop = (radius*radius*(lon_u - lon_l)*(np.sin(lat_u)-np.sin(lat_l)))

#Extra-Tropical weight
lat_extratrop = np.linspace(lat[0],lat[-1],num=len(lat),endpoint=True)
lat_extratrop[abs(lat_extratrop) < 23.5] = 90.0

lat_l = lat_extratrop/dtor
lat_u = (lat_extratrop + res_lat)/dtor

weights_extratrop = (radius*radius*(lon_u - lon_l)*(np.sin(lat_u)-np.sin(lat_l)))


###### NPP_NUPTAKE
average = np.ma.average(nee,axis=1,weights=weights)
average = np.ma.average(average,axis=1)
average_trop = np.ma.average(nee,axis=1,weights=weights_trop)
average_trop = np.ma.average(average_trop,axis=1)
average_extratrop = np.ma.average(nee,axis=1,weights=weights_extratrop)
average_extratrop = np.ma.average(average_extratrop,axis=1)
#####
average_stein1 = np.ma.average(nee_stein1,axis=1,weights=weights)
average_stein1 = np.ma.average(average_stein1,axis=1)
average_trop_stein1 = np.ma.average(nee_stein1,axis=1,weights=weights_trop)
average_trop_stein1 = np.ma.average(average_trop_stein1,axis=1)
average_extratrop_stein1 = np.ma.average(nee_stein1,axis=1,weights=weights_extratrop)
average_extratrop_stein1 = np.ma.average(average_extratrop_stein1,axis=1)
#####
average_stein2 = np.ma.average(nee_stein2,axis=1,weights=weights)
average_stein2 = np.ma.average(average_stein2,axis=1)
average_trop_stein2 = np.ma.average(nee_stein2,axis=1,weights=weights_trop)
average_trop_stein2 = np.ma.average(average_trop_stein2,axis=1)
average_extratrop_stein2 = np.ma.average(nee_stein2,axis=1,weights=weights_extratrop)
average_extratrop_stein2 = np.ma.average(average_extratrop_stein2,axis=1)
#####
average_stein3 = np.ma.average(nee_stein3,axis=1,weights=weights)
average_stein3 = np.ma.average(average_stein3,axis=1)
average_trop_stein3 = np.ma.average(nee_stein3,axis=1,weights=weights_trop)
average_trop_stein3 = np.ma.average(average_trop_stein3,axis=1)
average_extratrop_stein3 = np.ma.average(nee_stein3,axis=1,weights=weights_extratrop)
average_extratrop_stein3 = np.ma.average(average_extratrop_stein3,axis=1)
#####
average_stein4 = np.ma.average(nee_stein4,axis=1,weights=weights)
average_stein4 = np.ma.average(average_stein4,axis=1)
average_trop_stein4 = np.ma.average(nee_stein4,axis=1,weights=weights_trop)
average_trop_stein4 = np.ma.average(average_trop_stein4,axis=1)
average_extratrop_stein4 = np.ma.average(nee_stein4,axis=1,weights=weights_extratrop)
average_extratrop_stein4 = np.ma.average(average_extratrop_stein4,axis=1)

###### NUPTAKE
averagei = np.ma.average(neei,axis=1,weights=weights)
averagei = np.ma.average(averagei,axis=1)
averagei_trop = np.ma.average(neei,axis=1,weights=weights_trop)
averagei_trop = np.ma.average(averagei_trop,axis=1)
averagei_extratrop = np.ma.average(neei,axis=1,weights=weights_extratrop)
averagei_extratrop = np.ma.average(averagei_extratrop,axis=1)

#####
averagei_stein1 = np.ma.average(neei_stein1,axis=1,weights=weights)
averagei_stein1 = np.ma.average(averagei_stein1,axis=1)
averagei_trop_stein1 = np.ma.average(neei_stein1,axis=1,weights=weights_trop)
averagei_trop_stein1 = np.ma.average(averagei_trop_stein1,axis=1)
averagei_extratrop_stein1 = np.ma.average(neei,axis=1,weights=weights_extratrop)
averagei_extratrop_stein1 = np.ma.average(averagei_extratrop_stein1,axis=1)
#####
averagei_stein2 = np.ma.average(neei_stein2,axis=1,weights=weights)
averagei_stein2 = np.ma.average(averagei_stein2,axis=1)
averagei_trop_stein2 = np.ma.average(neei_stein2,axis=1,weights=weights_trop)
averagei_trop_stein2 = np.ma.average(averagei_trop_stein2,axis=1)
averagei_extratrop_stein2 = np.ma.average(neei_stein2,axis=1,weights=weights_extratrop)
averagei_extratrop_stein2 = np.ma.average(averagei_extratrop_stein2,axis=1)
#####
averagei_stein3 = np.ma.average(neei_stein3,axis=1,weights=weights)
averagei_stein3 = np.ma.average(averagei_stein3,axis=1)
averagei_trop_stein3 = np.ma.average(neei_stein3,axis=1,weights=weights_trop)
averagei_trop_stein3 = np.ma.average(averagei_trop_stein3,axis=1)
averagei_extratrop_stein3 = np.ma.average(neei_stein3,axis=1,weights=weights_extratrop)
averagei_extratrop_stein3 = np.ma.average(averagei_extratrop_stein3,axis=1)
#####
averagei_stein4 = np.ma.average(neei_stein4,axis=1,weights=weights)
averagei_stein4 = np.ma.average(averagei_stein4,axis=1)
averagei_trop_stein4 = np.ma.average(neei_stein4,axis=1,weights=weights_trop)
averagei_trop_stein4 = np.ma.average(averagei_trop_stein4,axis=1)
averagei_extratrop_stein4 = np.ma.average(neei_stein4,axis=1,weights=weights_extratrop)
averagei_extratrop_stein4 = np.ma.average(averagei_extratrop_stein4,axis=1)
######

###### PLANT_NDEMAND
averagej = np.ma.average(neej,axis=1,weights=weights)
averagej = np.ma.average(averagej,axis=1)
averagej_trop = np.ma.average(neej,axis=1,weights=weights_trop)
averagej_trop = np.ma.average(averagej_trop,axis=1)
averagej_extratrop = np.ma.average(neej,axis=1,weights=weights_extratrop)
averagej_extratrop = np.ma.average(averagej_extratrop,axis=1)
#####
averagej_stein1 = np.ma.average(neej_stein1,axis=1,weights=weights)
averagej_stein1 = np.ma.average(averagej_stein1,axis=1)
averagej_trop_stein1 = np.ma.average(neej_stein1,axis=1,weights=weights_trop)
averagej_trop_stein1 = np.ma.average(averagej_trop_stein1,axis=1)
averagej_extratrop_stein1 = np.ma.average(neej_stein1,axis=1,weights=weights_extratrop)
averagej_extratrop_stein1 = np.ma.average(averagej_extratrop_stein1,axis=1)
#####
averagej_stein2 = np.ma.average(neej_stein2,axis=1,weights=weights)
averagej_stein2 = np.ma.average(averagej_stein2,axis=1)
averagej_trop_stein2 = np.ma.average(neej_stein2,axis=1,weights=weights_trop)
averagej_trop_stein2 = np.ma.average(averagej_trop_stein2,axis=1)
averagej_extratrop_stein2 = np.ma.average(neej_stein2,axis=1,weights=weights_extratrop)
averagej_extratrop_stein2 = np.ma.average(averagej_extratrop_stein2,axis=1)
#####
averagej_stein3 = np.ma.average(neej_stein3,axis=1,weights=weights)
averagej_stein3 = np.ma.average(averagej_stein3,axis=1)
averagej_trop_stein3 = np.ma.average(neej_stein3,axis=1,weights=weights_trop)
averagej_trop_stein3 = np.ma.average(averagej_trop_stein3,axis=1)
averagej_extratrop_stein3 = np.ma.average(neej_stein3,axis=1,weights=weights_extratrop)
averagej_extratrop_stein3 = np.ma.average(averagej_extratrop_stein3,axis=1)
#####
averagej_stein4 = np.ma.average(neej_stein4,axis=1,weights=weights)
averagej_stein4 = np.ma.average(averagej_stein4,axis=1)
averagej_trop_stein4 = np.ma.average(neej_stein4,axis=1,weights=weights_trop)
averagej_trop_stein4 = np.ma.average(averagej_trop_stein4,axis=1)
averagej_extratrop_stein4 = np.ma.average(neej_stein4,axis=1,weights=weights_extratrop)
averagej_extratrop_stein4 = np.ma.average(averagej_extratrop_stein4,axis=1)
#####

###### NPP
averagek = np.ma.average(neek,axis=1,weights=weights)
averagek = np.ma.average(averagek,axis=1)
averagek_trop = np.ma.average(neek,axis=1,weights=weights_trop)
averagek_trop = np.ma.average(averagek_trop,axis=1)
averagek_extratrop = np.ma.average(neek,axis=1,weights=weights_extratrop)
averagek_extratrop = np.ma.average(averagek_extratrop,axis=1)
#####
averagek_stein1 = np.ma.average(neek_stein1,axis=1,weights=weights)
averagek_stein1 = np.ma.average(averagek_stein1,axis=1)
averagek_trop_stein1 = np.ma.average(neek_stein1,axis=1,weights=weights_trop)
averagek_trop_stein1 = np.ma.average(averagek_trop_stein1,axis=1)
averagek_extratrop_stein1 = np.ma.average(neek_stein1,axis=1,weights=weights_extratrop)
averagek_extratrop_stein1 = np.ma.average(averagek_extratrop_stein1,axis=1)
#####
averagek_stein2 = np.ma.average(neek_stein2,axis=1,weights=weights)
averagek_stein2 = np.ma.average(averagek_stein2,axis=1)
averagek_trop_stein2 = np.ma.average(neek_stein2,axis=1,weights=weights_trop)
averagek_trop_stein2 = np.ma.average(averagek_trop_stein2,axis=1)
averagek_extratrop_stein2 = np.ma.average(neek_stein2,axis=1,weights=weights_extratrop)
averagek_extratrop_stein2 = np.ma.average(averagek_extratrop_stein2,axis=1)
#####
averagek_stein3 = np.ma.average(neek_stein3,axis=1,weights=weights)
averagek_stein3 = np.ma.average(averagek_stein3,axis=1)
averagek_trop_stein3 = np.ma.average(neek_stein3,axis=1,weights=weights_trop)
averagek_trop_stein3 = np.ma.average(averagek_trop_stein3,axis=1)
averagek_extratrop_stein3 = np.ma.average(neek_stein3,axis=1,weights=weights_extratrop)
averagek_extratrop_stein3 = np.ma.average(averagek_extratrop_stein3,axis=1)
#####
averagek_stein4 = np.ma.average(neek_stein4,axis=1,weights=weights)
averagek_stein4 = np.ma.average(averagek_stein4,axis=1)
averagek_trop_stein4 = np.ma.average(neek_stein4,axis=1,weights=weights_trop)
averagek_trop_stein4 = np.ma.average(averagek_trop_stein4,axis=1)
averagek_extratrop_stein4 = np.ma.average(neek_stein4,axis=1,weights=weights_extratrop)
averagek_extratrop_stein4 = np.ma.average(averagek_extratrop_stein4,axis=1)
#####

#plt.plot(averagej,'k',label='PLANT_NDEMAND')
#plt.plot(averagei,'b',label='NUPTAKE')
#plt.legend(loc=4)
#plt.savefig('noffer_vs_demand.png')
#plt.show()
#sys.exit()


slope_map = np.zeros((96,144))
slopei_map = np.zeros((96,144))
slopej_map = np.zeros((96,144))
slopek_map = np.zeros((96,144))

nee_av = nee
nee_av_stein1 = nee_stein1
nee_av_stein2 = nee_stein2
nee_av_stein3 = nee_stein3
nee_av_stein4 = nee_stein4

neei_av = neei
neei_av_stein1 = neei_stein1
neei_av_stein2 = neei_stein2
neei_av_stein3 = neei_stein3
neei_av_stein4 = neei_stein4

neej_av = neej
neej_av_stein1 = neej_stein1
neej_av_stein2 = neej_stein2
neej_av_stein3 = neej_stein3
neej_av_stein4 = neej_stein4

neek_av = neek
neek_av_stein1 = neek_stein1
neek_av_stein2 = neek_stein2
neek_av_stein3 = neek_stein3
neek_av_stein4 = neek_stein4

#smoothing the whole matrix 
for i in xrange(96):
  for j in xrange(144):
     nee_av[:,i,j] = smooth1(nee[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein1[:,i,j] = smooth1(nee_stein1[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein2[:,i,j] = smooth1(nee_stein2[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein3[:,i,j] = smooth1(nee_stein3[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein4[:,i,j] = smooth1(nee_stein4[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

     neei_av[:,i,j] = smooth1(neei[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein1[:,i,j] = smooth1(neei_stein1[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein2[:,i,j] = smooth1(neei_stein2[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein3[:,i,j] = smooth1(neei_stein3[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein4[:,i,j] = smooth1(neei_stein4[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

     neej_av[:,i,j] = smooth1(neej[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein1[:,i,j] = smooth1(neej_stein1[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein2[:,i,j] = smooth1(neej_stein2[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein3[:,i,j] = smooth1(neej_stein3[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein4[:,i,j] = smooth1(neej_stein4[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

     neek_av[:,i,j] = smooth1(neek[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein1[:,i,j] = smooth1(neek_stein1[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein2[:,i,j] = smooth1(neek_stein2[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein3[:,i,j] = smooth1(neek_stein3[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein4[:,i,j] = smooth1(neek_stein4[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

time = time/365.
time_future = time_future/365.

#for i in xrange(96):
#  for j in xrange(144):
     #NPP_NUPTAKE
#     slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av[:,i,j])
#     slope_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av_stein1[:,i,j])
#     slope_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av_stein2[:,i,j])
#     slope_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av_stein3[:,i,j])
#     slope_stein4, intercept, r_value, p_value, std_err = stats.linregress(time_future[:],nee_av_stein4[:,i,j])
     #NUPTAKE
#     slopei, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av[:,i,j])
#     slopei_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av_stein1[:,i,j])
#     slopei_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av_stein2[:,i,j])
#     slopei_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av_stein3[:,i,j])
#     slopei_stein4, intercept, r_value, p_value, std_err = stats.linregress(time_future[:],neei_av_stein4[:,i,j])
     #PLANT_NDEMAND
#     slopej, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av[:,i,j])
#     slopej_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av_stein1[:,i,j])
#     slopej_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av_stein2[:,i,j])
#     slopej_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av_stein3[:,i,j])
#     slopej_stein4, intercept, r_value, p_value, std_err = stats.linregress(time_future[:],neej_av_stein4[:,i,j])
     #NPP
#     slopek, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av[:,i,j])
#     slopek_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein1[:,i,j])
#     slopek_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein2[:,i,j])
#     slopek_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein3[:,i,j])
#     slopek_stein4, intercept, r_value, p_value, std_err = stats.linregress(time_future[:],neek_av_stein4[:,i,j])



#     if((slopej > 0.0 and slopei > 0.0 and abs(slopej) > abs(slopei))): 
        #slope_map[i,j] =  (slopei/slopej) - 1.
#        slope_map[i,j] =  (slopei/slopej)
#     slope_map[i,j] =  slopek
     #if((slopej < 0.0 and slopei < 0.0 and abs(slopej) > abs(slopei))): 
      #  slope_map[i,j] =   (slopei/slopej)
        #slope_map[i,j] =  1. - (slopei/slopej)
     #if(slope > 0.0):
     #   slope_map[i,j] = slope
     #if(slope < 0.0):
     #   slope_map[i,j] = slope
     #if(slope == 0.0):
     #   slope_map[i,j] = slope


#plt.imshow(slope_map)
#plt.colorbar()
#plt.show()

#plt.hist(np.nan_to_num(slope_map.ravel()), bins='auto')
#plt.show()

#print('min',np.nanmin(slope_map),'max', np.nanmax(slope_map))
#print('mean',np.nanmean(slope_map),'std', np.nanstd(slope_map))

#slope_map = slope_map/abs(np.max(slope_map))
#slope_map = 1. - slope_map

#print(np.min(slope_map), np.max(slope_map))

#lon = f.variables['lon'][:]
#slope_map,lon = shiftgrid(180., slope_map, lon, start=False)


#pft_map_nat = surf.variables['PCT_NAT_PFT'][:]
#lon = f.variables['lon'][:]
#pft_map_nat,lon = shiftgrid(180., pft_map_nat, lon, start=False)

#slope_map = ma.array(slope_map,mask=[pft_map_nat[0,:,:]>90.])



#m = Basemap(projection='robin', lon_0=0.,resolution='l')
#x,y = np.meshgrid(lon, lat) 
#X,Y = m(x, y)

#x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*20)
#y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*30)

#x2,y2 = np.meshgrid(x2,y2)
#X2,Y2 = m(x2, y2)

#data2 = interp(slope_map,x[0],y[:,0],x2,y2,order=1)

#mdata = maskoceans(x2, y2, data2,resolution='h',grid=1.25,inlands=True)



#for i in xrange(1):
#   fig = plt.figure(figsize=(48, 48)) 
#   m.drawmapboundary(fill_color='white', zorder=-1)
#   m.fillcontinents(color='0.8', lake_color='white', zorder=0)
 
#   m.drawcoastlines(color='0.6', linewidth=0.5)
#   m.drawcountries(color='0.6', linewidth=0.5)
#   m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1],    dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')
#   m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')

 
   #PLOT ABSOLUTE
#   levels = np.linspace(0.,100.,21)
#   vmin,vmax = (0,100)
#   cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.jet,extend='both')
   
   #PLOT DIFFERENCE
#   vmin,vmax = (0.25,1.)
   #vmin,vmax = (np.min(slope_map), np.max(slope_map))
#   levels = np.linspace(vmin,vmax,16,endpoint=True)   
#   cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.RdYlGn_r,extend='both')
   

#   plt.tight_layout()
#   cbar = m.colorbar(cs,location='bottom',pad='10%',ticks=np.linspace(vmin,vmax,3),format='%.1f')
#   cbar.ax.set_xticklabels(['Low risk','Medium risk','High risk'])
   #cbar.ax.get_yaxis().labelpad = 60
#   cbar.ax.get_xaxis().labelpad = 60
   #cbar.ax.set_ylabel('EM (%)', rotation=270)
#   cbar.ax.set_xlabel('Nitrogen limitation', rotation=0,color='black', size=78)
   #cbar.ax.set_xlabel('ECM tree basal area (%)', rotation=0,color='black', size=78)
#   cbar.solids.set_edgecolor("face")
   #cbar.set_clim(0.0,100)
#   cbar.set_clim(vmin,vmax)
   #plt.title(r'Sulman et al. (2019) - Shi et al. (2016)')
#   cbar.ax.tick_params(labelsize='xx-large')
   #plt.savefig('offer_vs_demand_clm5.png',bbox_inches="tight")
   #plt.savefig('diff_em_tot_3.pdf',bbox_inches="tight",dpi=300)
   #plt.savefig('ecm_orig_shi_1p9x2p5.png',bbox_inches="tight",dpi=300)
   #plt.show()

#sys.exit()

y_av = smooth1(average[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
tropic_land = 0.36
y_av_trop = smooth1(average_trop[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av_extratrop = smooth1(average_extratrop[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
x_av = smooth1(time,120)
y_av_stein1 = smooth1(average_stein1[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av_trop_stein1 = smooth1(average_trop_stein1[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av_extratrop_stein1 = smooth1(average_extratrop_stein1[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av_stein2 = smooth1(average_stein2[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av_trop_stein2 = smooth1(average_trop_stein2[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av_extratrop_stein2 = smooth1(average_extratrop_stein2[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av_stein3 = smooth1(average_stein3[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av_trop_stein3 = smooth1(average_trop_stein3[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av_extratrop_stein3 = smooth1(average_extratrop_stein3[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av_stein4 = smooth1(average_stein4[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av_trop_stein4 = smooth1(average_trop_stein4[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av_extratrop_stein4 = smooth1(average_extratrop_stein4[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)

average1 = np.ma.average(nee_stein1,axis=1,weights=weights)
average1 = np.ma.average(average1,axis=1)

y_av1 = smooth1(averagek[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_trop = smooth1(averagek_trop[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av1_extratrop = smooth1(averagek_extratrop[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av1_stein1 = smooth1(averagek_stein1[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_trop_stein1 = smooth1(averagek_trop_stein1[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av1_extratrop_stein1 = smooth1(averagek_extratrop_stein1[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av1_stein2 = smooth1(averagek_stein2[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_trop_stein2 = smooth1(averagek_trop_stein2[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av1_extratrop_stein2 = smooth1(averagek_extratrop_stein2[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av1_stein3 = smooth1(averagek_stein3[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_trop_stein3 = smooth1(averagek_trop_stein3[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av1_extratrop_stein3 = smooth1(averagek_extratrop_stein3[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av1_stein4 = smooth1(averagek_stein4[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_trop_stein4 = smooth1(averagek_trop_stein4[:],12)*(1)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av1_extratrop_stein4 = smooth1(averagek_extratrop_stein4[:],12)*(1)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)

average2 = np.ma.average(nee_stein2,axis=1,weights=weights)
average2 = np.ma.average(average2,axis=1)

y_av2 = smooth1(averagei[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_trop = smooth1(averagei_trop[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av2_extratrop = smooth1(averagei_extratrop[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)

#######
y_av2_stein1 = smooth1(averagei_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_trop_stein1 = smooth1(averagei_trop_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av2_extratrop_stein1 = smooth1(averagei_extratrop_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av2_stein2 = smooth1(averagei_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_trop_stein2 = smooth1(averagei_trop_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av2_extratrop_stein2 = smooth1(averagei_extratrop_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av2_stein3 = smooth1(averagei_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_trop_stein3 = smooth1(averagei_trop_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av2_extratrop_stein3 = smooth1(averagei_extratrop_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av2_stein4 = smooth1(averagei_stein4[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_trop_stein4 = smooth1(averagei_trop_stein4[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av2_extratrop_stein4 = smooth1(averagei_extratrop_stein4[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)

average3 = np.ma.average(nee_stein3,axis=1,weights=weights)
average3 = np.ma.average(average3,axis=1)

y_av3 = smooth1(averagej[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_trop = smooth1(averagej_trop[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av3_extratrop = smooth1(averagej_extratrop[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
#####
y_av3_stein1 = smooth1(averagej_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_trop_stein1 = smooth1(averagej_trop_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av3_extratrop_stein1 = smooth1(averagej_extratrop_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av3_stein2 = smooth1(averagej_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_trop_stein2 = smooth1(averagej_trop_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av3_extratrop_stein2 = smooth1(averagej_extratrop_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av3_stein3 = smooth1(averagej_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_trop_stein3 = smooth1(averagej_trop_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av3_extratrop_stein3 = smooth1(averagej_extratrop_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)
y_av3_stein4 = smooth1(averagej_stein4[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_trop_stein4 = smooth1(averagej_trop_stein4[:],12)*(1000)*(365*24*60*60)*148847000*(tropic_land)*(1000*1000)/(10**15)
y_av3_extratrop_stein4 = smooth1(averagej_extratrop_stein4[:],12)*(1000)*(365*24*60*60)*148847000*(1.-tropic_land)*(1000*1000)/(10**15)

mean = np.mean(y_av[0:30*12*50])
mean1 = np.mean(y_av1[0:30*12*50])
mean2 = np.mean(y_av2[0:30*12*50])
mean3 = np.mean(y_av3[0:30*12*50])



time = time*365
time_future = time_future*365


start = 0
end = 30*12*161

start_future = 0 + end
end_future = 30*12*55 + end + (30*12*5)
#xtick = np.arange(start, end,step = 12*25*30)
xtick = np.arange(start, end_future,step = 12*25*30)


#plt.style.use('ggplot')
plt.style.use('seaborn-whitegrid')
SIZE = 13
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)  # # size of the figure title


fig = plt.figure()
grid = plt.GridSpec(1,5) # 1 row 4 cols
ax1 = plt.subplot(grid[0, :4]) # top left 
ax2 = plt.subplot(grid[0, 4]) # top right  

slope_av = []
slope_av1 = []
#for i in xrange(50):
#   count = 38*i
#   slope, intercept, r_value, p_value, std_err = stats.linregress(time[0+count:38+count],y_av[0+count:38+count])
#   slope_av.append(slope)
#   slope, intercept, r_value, p_value, std_err = stats.linregress(time[0+count:38+count],y_av1[0+count:38+count])
#   slope_av1.append(slope)

#slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av_stein3[:])
#print('NPP_NUPTAKE',slope,100.*slope/np.mean(y_av_stein3[:]))
#print('NPP_NUPTAKE','mean',np.mean(y_av_stein3[:]),'std',np.std(y_av_stein3[:]))

#slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av1_stein3[:])
#print('NPP',slope,100.*slope/np.mean(y_av1_stein3[:]))
#print('NPP','mean',np.mean(y_av1_stein3[:]),'std',np.std(y_av1_stein3[:]))

#slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av2_stein3[:])
#print('NUPTAKE',slope,100.*slope/np.mean(y_av2_stein3[:]))
#print('NUPTAKE','mean',np.mean(y_av2_stein3[:]),'std',np.std(y_av2_stein3[:]))

#slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av3_stein3[:])
#print('PLANT_DEMAND',slope,100.*slope/np.mean(y_av3_stein3[:]))
#print('PLANT_DEMAND','mean',np.mean(y_av3_stein3[:]),'std',np.std(y_av3_stein3[:]))

##### AV IS NPP_NUPTAKE
#ax1.plot(time,y_av,'r',label='Global')
#ax1.plot(time,y_av_trop,'g',label='Tropical')
#ax1.plot(time,y_av_extratrop,'b',label='Extra-Tropical')
#ax1.plot(time,y_av_stein1,'r',alpha=0.3)
#ax1.plot(time,y_av_trop_stein1,'g',alpha=0.3)
#ax1.plot(time,y_av_extratrop_stein1,'b',alpha=0.3)
#ax1.plot(time,y_av_stein2,'r',alpha=0.4)
#ax1.plot(time,y_av_trop_stein2,'g',alpha=0.4)
#ax1.plot(time,y_av_extratrop_stein2,'b',alpha=0.4)
#ax1.plot(time,y_av_stein3,'r',alpha=0.5)
#ax1.plot(time,y_av_trop_stein3,'g',alpha=0.5)
#ax1.plot(time,y_av_extratrop_stein3,'b',alpha=0.5)
#ax1.plot(time_future[1:732] + end + (30*12*5),y_av_stein4[1:732] + 4.,'r--')
#ax1.plot(time_future[1:732] + end + (30*12*5),y_av_trop_stein4[1:732] + 2.,'g--')
#ax1.plot(time_future[1:732] + end + (30*12*5),y_av_extratrop_stein4[1:732] + 2.,'b--')

##### AV1 IS NUPTAKE
ax1.plot(time[0:1930],y_av3[0:1930],'r',label='Global')
ax1.plot(time[0:1930],y_av3_trop[0:1930],'g',label='Tropical')
ax1.plot(time[0:1930],y_av3_extratrop[0:1930],'b',label='Extra-Tropical')
ax1.plot(time[0:1930],y_av3_stein1[0:1930],'r',alpha=0.3)
ax1.plot(time[0:1930],y_av3_trop_stein1[0:1930],'g',alpha=0.3)
ax1.plot(time[0:1930],y_av3_extratrop_stein1[0:1930],'b',alpha=0.3)
#ax1.plot(time[0:1930],y_av3_stein2[0:1930],'r',alpha=0.4)
#ax1.plot(time[0:1930],y_av3_trop_stein2[0:1930],'g',alpha=0.4)
#ax1.plot(time[0:1930],y_av2_extratrop_stein2[0:1930],'b',alpha=0.4)
ax1.plot(time[0:1930],y_av3_stein3[0:1930],'r',alpha=0.5)
ax1.plot(time[0:1930],y_av3_trop_stein3[0:1930],'g',alpha=0.5)
ax1.plot(time[0:1930],y_av3_extratrop_stein3[0:1930],'b',alpha=0.5)
ax1.plot(time_future[3:732] + end + (30*12*5),y_av3_stein4[3:732] ,'r--')
ax1.plot(time_future[3:732] + end + (30*12*5),y_av3_trop_stein4[3:732]+118. ,'g--')
ax1.plot(time_future[3:732] + end + (30*12*5),y_av3_extratrop_stein4[3:732] ,'b--')


ax1.legend(loc='best',fontsize ="small",ncol=2,borderpad=0.)
#plt.ylim(0,16)
ax1.set_ylim(0.,3000.)
#plt.xlim(start,end)
#ax1.set_xlim(start,end)
#plt.xticks(np.arange(start, end,step = 12*25*30), ('1850','1875','1900','1925','1950','1975','2000'))
end_future = 30*12*55 + end + (30*12*5)
#xtick = np.arange(start, end,step = 12*25*30)
xtick = np.arange(start, end_future + (30*12*25), step = 12*50*30)

ax1.set_xlim(start,end_future + (30*12*25))
ax1.set_xticks(xtick)
#ax1.set_xticklabels(['1850','1875','1900','1925','1950','1975','2000'])
ax1.set_xticklabels(np.arange(1850, 2070 + 25, step = 50),fontdict={'fontsize':12})
ax1.tick_params(axis='x', pad=8)
#plt.ylabel(r'NCEM (TgN.yr$^{-1}$)')
ax1.set_ylabel(r'N Demand (TgN.yr$^{-1}$)')
#plt.xlabel('year')
ax1.set_xlabel('year')
#best location
ax1.legend(loc=0,fontsize='x-large')

my_pal1 = {"red","darksalmon"}
my_pal2 = {"green","lightgreen"}
my_pal3 = {"blue","lightskyblue"}


#vector1 = [y_av,y_av_stein4+4.]
#vector2 = [y_av_trop,y_av_trop_stein4+2.]
#vector3 = [y_av_extratrop,y_av_extratrop_stein4+2.]

#sns.boxplot(data = vector1,palette=my_pal1,ax=ax2)
#sns.boxplot(data = vector2,palette=my_pal2,ax=ax2)
#sns.boxplot(data = vector3,palette=my_pal3,ax=ax2)

vector1 = [y_av3[0:1930],y_av3_stein4[3:732]]
vector2 = [y_av3_trop[0:1930],y_av3_trop_stein4[3:732]+118.]
vector3 = [y_av3_extratrop[0:1930],y_av3_extratrop_stein4[3:732]]

sns.boxplot(data = vector1,palette=my_pal1,ax=ax2)
sns.boxplot(data = vector2,palette=my_pal2,ax=ax2)
sns.boxplot(data = vector3,palette=my_pal3,ax=ax2)


ax2.set_ylim(0.,3000.)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_yticklabels('')
#ax2.set_yticks([])
ax2.set_xticklabels(['Hist','Fut'])
#plt.title('US - UMB')
plt.savefig('regional_PLANT_NDEMAND.png',bbox_inches="tight")
plt.show()


sys.exit()

