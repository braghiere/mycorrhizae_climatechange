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
#plt.style.use('ggplot')
#plt.style.use('fivethirtyeight')
plt.style.use('seaborn-whitegrid')
SIZE = 13
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)  # # size of the figure title

var = 'AM'

f = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50Bgc/IHistClm50Bgc.'+ var +'.nc','r')
f1 = Dataset('/home/renato/datasets/figures/fig_larger/Fig4/IHistClm50BgcSulman_v2.clm2.h0.'+ var +'.nc','r')
f2 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.'+ var +'.nc','r')
f3 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.'+ var +'.nc','r')

surf = Dataset('/home/renato/Steindinger/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190304.nc','r')
params = Dataset('/home/renato/Steindinger/clm5_params.c171117.nc','r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time'][:]
#var = 'NECM'
nee = f.variables[var]
unit = nee.units
long_name = nee.long_name
nee = f.variables[var][:]
nee_stein1 = f1.variables[var][:]
nee_stein2 = f2.variables[var][:]
nee_stein3 = f3.variables[var][:]

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

lat_l = lat/dtor
lat_u = (lat + res_lat)/dtor

weights = (radius*radius*(lon_u - lon_l)*(np.sin(lat_u)-np.sin(lat_l)))


tgnyr = (365*24*60*60)*148000847*(1000*1000)/(10**12)
pgcyr = (1000)*(365*24*60*60)*148000847*(1000*1000)/(10**15)

average = np.ma.average(nee,axis=1,weights=weights)
average = np.ma.average(average,axis=1)

#print(np.shape(average[:]))
y_av = smooth1(average[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
x_av = smooth1(time,120)

average1 = np.ma.average(nee_stein1,axis=1,weights=weights)
average1 = np.ma.average(average1,axis=1)

y_av1 = smooth1(average1[:],12)*(1000)*(365*24*60*60)*148000847*(1000*1000)/(10**15)

average2 = np.ma.average(nee_stein2,axis=1,weights=weights)
average2 = np.ma.average(average2,axis=1)

y_av2 = smooth1(average2[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

average3 = np.ma.average(nee_stein3,axis=1,weights=weights)
average3 = np.ma.average(average3,axis=1)

y_av3 = smooth1(average3[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

mean = np.mean(y_av[0:30*12*50])
mean1 = np.mean(y_av1[0:30*12*50])
mean2 = np.mean(y_av2[0:30*12*50])
mean3 = np.mean(y_av3[0:30*12*50])

print(np.shape(y_av))
print(np.shape(y_av1))
print(np.shape(y_av2))
print(np.shape(y_av3))
print('CLM5=',mean)
print('Stein=',mean1)
print('Sulman=',mean2)
print('Soudzi=',mean3)
print(time)
start = 0
end = 30*12*161

xtick = np.arange(start, end,step = 12*25*30)

fig = plt.figure()
grid = plt.GridSpec(1,5) # 1 row 4 cols
ax1 = plt.subplot(grid[0, :4]) # top left 
ax2 = plt.subplot(grid[0, 4]) # top right  

ax1.plot(time[5:1932],y_av[5:1932],'k',label='CLM5.0')
ax1.plot(time[5:1932],y_av1[5:1932],'r',label='Sulman et al. (2019)')
ax1.plot(time[5:1932],y_av2[5:1932],'b',label='Steidinger et al. (2019)')
ax1.plot(time[5:1932],y_av3[5:1932],'g',label='Soudzilovskaia et al. (2019)')
ax1.legend(loc='best',fontsize ="small",ncol=2,mode="expand",
           borderpad=0., borderaxespad=0.5)
#plt.ylim(0,16)
ax1.set_ylim(0,16)

ax1.set_xlim(start,end)
#plt.xticks(np.arange(start, end,step = 12*25*30), ('1850','1875','1900','1925','1950','1975','2000'))
ax1.set_xticks(xtick)
ax1.set_xticklabels(['1850','1875','1900','1925','1950','1975','2000'])
ax1.tick_params(axis='x', pad=10)
#plt.ylabel(r'NCEM (TgN.yr$^{-1}$)')
#ax1.set_ylabel(r'NRETRANS (TgN.yr$^{-1}$)')
ax1.set_ylabel(r'N Uptake by ECM (TgN.yr$^{-1}$)')
#plt.xlabel('year')
plt.tight_layout()
ax1.set_xlabel('year')

sns.boxplot(data = y_av,color='k',ax=ax2)
sns.boxplot(data = y_av1[0:1932],color='r',ax=ax2)
sns.boxplot(data = y_av2,color='b',ax=ax2)
sns.boxplot(data = y_av3,color='g',ax=ax2)
ax2.set_ylim(0,16)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_yticklabels('')
#ax2.set_yticks([])
ax2.set_xticklabels('')
#plt.title('US - UMB')
plt.tight_layout()
plt.savefig('/home/renato/datasets/figures/fig_larger/Fig4/NECM_timeseries_boxplot_nogray_v2.png',bbox_inches="tight")
plt.show()


sys.exit()


