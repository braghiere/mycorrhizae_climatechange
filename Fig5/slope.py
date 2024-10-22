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
fl = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50Bgc/IHistClm50Bgc.NUPTAKE_NPP_FRACTION.nc','r')

f1 = Dataset('/home/renato/datasets/figures/fig_larger/Fig4/IHistClm50BgcSulman_v2.NPP_NUPTAKE.nc','r')
f1i = Dataset('/home/renato/datasets/figures/fig_larger/Fig4/IHistClm50BgcSulman_v2.NUPTAKE.nc','r')
f1j = Dataset('/home/renato/datasets/figures/fig_larger/Fig4/IHistClm50BgcSulman_v2.PLANT_NDEMAND.nc','r')
f1k = Dataset('/home/renato/datasets/figures/fig_larger/Fig4/IHistClm50BgcSulman_v2.NPP.nc','r')
f1l = Dataset('/home/renato/datasets/figures/fig_larger/Fig4/IHistClm50BgcSulman_v2.NUPTAKE_NPP_FRACTION.nc','r')

f2 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.NPP_NUPTAKE.nc','r')
f2i = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.NUPTAKE.nc','r')
f2j = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.PLANT_NDEMAND.nc','r')
f2k = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSteidinger/IHistClm50BgcSteidinger.NPP.nc','r')

f3 = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.NPP_NUPTAKE.nc','r')
f3i = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.NUPTAKE.nc','r')
f3j = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.PLANT_NDEMAND.nc','r')
f3k = Dataset('/home/renato/datasets/IHistClm50Bgc_outputs/IHistClm50BgcSoudzi/IHistClm50BgcSoudzi.NPP.nc','r')

surf = Dataset('/home/renato/Steindinger/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190304.nc','r')
params = Dataset('/home/renato/Steindinger/clm5_params.c171117.nc','r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time'][:]
print time
var = 'NPP_NUPTAKE'
nee = f1.variables[var]
unit = nee.units
long_name = nee.long_name
nee_stein1 = f1.variables[var][:]
nee_stein2 = f2.variables[var][:]
nee_stein3 = f3.variables[var][:]
var = 'NPP_NUPTAKE'
nee = f.variables[var][:]
nee_stein1 = f1.variables[var][:]
var = 'NUPTAKE'
neei = fi.variables[var][:]
neei_stein1 = f1i.variables[var][:]
neei_stein2 = f2i.variables[var][:]
neei_stein3 = f3i.variables[var][:]
var = 'PLANT_NDEMAND'
neej = fj.variables[var][:]
neej_stein1 = f1j.variables[var][:]
neej_stein2 = f2j.variables[var][:]
neej_stein3 = f3j.variables[var][:]
var = 'NPP'
neek = fk.variables[var][:]
neek_stein1 = f1k.variables[var][:]
neek_stein2 = f2k.variables[var][:]
neek_stein3 = f3k.variables[var][:]

var = 'NUPTAKE_NPP_FRACTION'
neel = fl.variables[var][:]
neel_stein1 = f1l.variables[var][:]
#neel_stein2 = f2k.variables[var][:]
#neel_stein3 = f3k.variables[var][:]


neel[neel>1.0]=1.0
neel[neel<0.0]=0.0
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

average = np.ma.average(nee,axis=1,weights=weights)
average = np.ma.average(average,axis=1)
average_stein1 = np.ma.average(nee_stein1,axis=1,weights=weights)
average_stein1 = np.ma.average(average_stein1,axis=1)
average_stein2 = np.ma.average(nee_stein2,axis=1,weights=weights)
average_stein2 = np.ma.average(average_stein2,axis=1)
average_stein3 = np.ma.average(nee_stein3,axis=1,weights=weights)
average_stein3 = np.ma.average(average_stein3,axis=1)

averagei = np.ma.average(neei,axis=1,weights=weights)
averagei = np.ma.average(averagei,axis=1)
averagei_stein1 = np.ma.average(neei_stein1,axis=1,weights=weights)
averagei_stein1 = np.ma.average(averagei_stein1,axis=1)
averagei_stein2 = np.ma.average(neei_stein2,axis=1,weights=weights)
averagei_stein2 = np.ma.average(averagei_stein2,axis=1)
averagei_stein3 = np.ma.average(neei_stein3,axis=1,weights=weights)
averagei_stein3 = np.ma.average(averagei_stein3,axis=1)

averagej = np.ma.average(neej,axis=1,weights=weights)
averagej = np.ma.average(averagej,axis=1)
averagej_stein1 = np.ma.average(neej_stein1,axis=1,weights=weights)
averagej_stein1 = np.ma.average(averagej_stein1,axis=1)
averagej_stein2 = np.ma.average(neej_stein2,axis=1,weights=weights)
averagej_stein2 = np.ma.average(averagej_stein2,axis=1)
averagej_stein3 = np.ma.average(neej_stein3,axis=1,weights=weights)
averagej_stein3 = np.ma.average(averagej_stein3,axis=1)

averagek = np.ma.average(neek,axis=1,weights=weights)
averagek = np.ma.average(averagek,axis=1)
averagek_stein1 = np.ma.average(neek_stein1,axis=1,weights=weights)
averagek_stein1 = np.ma.average(averagek_stein1,axis=1)
averagek_stein2 = np.ma.average(neek_stein2,axis=1,weights=weights)
averagek_stein2 = np.ma.average(averagek_stein2,axis=1)
averagek_stein3 = np.ma.average(neek_stein3,axis=1,weights=weights)
averagek_stein3 = np.ma.average(averagek_stein3,axis=1)

averagel = np.ma.average(neel,axis=1,weights=weights)
averagel = np.ma.average(averagel,axis=1)
averagel_stein1 = np.ma.average(neel_stein1,axis=1,weights=weights)
averagel_stein1 = np.ma.average(averagel_stein1,axis=1)


slope_map = np.zeros((96,144))
slopei_map = np.zeros((96,144))
slopej_map = np.zeros((96,144))
slopek_map = np.zeros((96,144))
slopel_map = np.zeros((96,144))

nee_av = nee
nee_av_stein1 = nee_stein1
nee_av_stein2 = nee_stein2
nee_av_stein3 = nee_stein3

neei_av = neei
neei_av_stein1 = neei_stein1
neei_av_stein2 = neei_stein2
neei_av_stein3 = neei_stein3

neej_av = neej
neej_av_stein1 = neej_stein1
neej_av_stein2 = neej_stein2
neej_av_stein3 = neej_stein3

neek_av = neek
neek_av_stein1 = neek_stein1
neek_av_stein2 = neek_stein2
neek_av_stein3 = neek_stein3

neel_av = neel
neel_av_stein1 = neel_stein1

#smoothing the whole matrix 
for i in xrange(96):
  for j in xrange(144):
     nee_av[:,i,j] = smooth1(nee[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein1[:,i,j] = smooth1(nee_stein1[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein2[:,i,j] = smooth1(nee_stein2[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     nee_av_stein3[:,i,j] = smooth1(nee_stein3[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

     neei_av[:,i,j] = smooth1(neei[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein1[:,i,j] = smooth1(neei_stein1[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein2[:,i,j] = smooth1(neei_stein2[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neei_av_stein3[:,i,j] = smooth1(neei_stein3[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

     neej_av[:,i,j] = smooth1(neej[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein1[:,i,j] = smooth1(neej_stein1[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein2[:,i,j] = smooth1(neej_stein2[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neej_av_stein3[:,i,j] = smooth1(neej_stein3[:,i,j],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

     neek_av[:,i,j] = smooth1(neek[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein1[:,i,j] = smooth1(neek_stein1[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein2[:,i,j] = smooth1(neek_stein2[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
     neek_av_stein3[:,i,j] = smooth1(neek_stein3[:,i,j],12)*(365*24*60*60)*148847000*(1000*1000)/(10**15)



     neel_av[:,i,j] = smooth1(neel[:,i,j],12)*100.
     neel_av_stein1[:,i,j] = smooth1(neel_stein1[:,i,j],12)*100.

time = time/365.

for i in xrange(96):
  for j in xrange(144):
     #NPP_NUPTAKE
     slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av[:,i,j])
     slope_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av_stein1[:,i,j])
     slope_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av_stein2[:,i,j])
     slope_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],nee_av_stein3[:,i,j])
     #NUPTAKE
     slopei, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av[:,i,j])
     slopei_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av_stein1[:,i,j])
     slopei_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av_stein2[:,i,j])
     slopei_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neei_av_stein3[:,i,j])
     #PLANT_NDEMAND
     slopej, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av[:,i,j])
     slopej_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av_stein1[:,i,j])
     slopej_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av_stein2[:,i,j])
     slopej_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neej_av_stein3[:,i,j])
     #NPP
     slopek, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av[:,i,j])
     slopek_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein1[:,i,j])
     slopek_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein2[:,i,j])
     slopek_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein3[:,i,j])
     #NUPTAKE_NPP_FRACTION
     slopel, intercept, r_value, p_value, std_err = stats.linregress(time[:],neel_av[:,i,j])
     slopel_stein1, intercept, r_value, p_value, std_err = stats.linregress(time[:],neel_av_stein1[:,i,j])
     #slopek_stein2, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein2[:,i,j])
     #slopek_stein3, intercept, r_value, p_value, std_err = stats.linregress(time[:],neek_av_stein3[:,i,j])


     #if((slopej > 0.0 and slopei > 0.0 and abs(slopej) > abs(slopei))): 
        #slope_map[i,j] =  (slopei/slopej) - 1.
     slope_map[i,j] =  slopel
     #slope_map[i,j] =  slopek
     #if((slopej < 0.0 and slopei < 0.0 and abs(slopej) > abs(slopei))): 
      #  slope_map[i,j] =   (slopei/slopej)
        #slope_map[i,j] =  1. - (slopei/slopej)
     #if(slope > 0.0):
     #   slope_map[i,j] = slope
     #if(slope < 0.0):
     #   slope_map[i,j] = slope
     #if(slope == 0.0):
     #   slope_map[i,j] = slope


plt.imshow(slope_map)
plt.colorbar()
plt.show()

#plt.hist(np.nan_to_num(slope_map.ravel()), bins='auto')
#plt.show()

print('min',np.nanmin(slope_map),'max', np.nanmax(slope_map))
print('mean',np.nanmean(slope_map),'std', np.nanstd(slope_map))
print('quartile_25%',np.nanquantile(slope_map,0.25),'quartile_75%', np.nanquantile(slope_map,0.75))



#slope_map = slope_map/abs(np.nanmax(slope_map))
slope_map = 1. - slope_map

print(slope_map)

print(np.min(slope_map), np.max(slope_map))

lon = f.variables['lon'][:]
slope_map,lon = shiftgrid(180., slope_map, lon, start=False)


pft_map_nat = surf.variables['PCT_NAT_PFT'][:]
lon = f.variables['lon'][:]
pft_map_nat,lon = shiftgrid(180., pft_map_nat, lon, start=False)

slope_map = ma.array(slope_map,mask=[pft_map_nat[0,:,:]>90.])



m = Basemap(projection='robin', lon_0=0.,resolution='l')
x,y = np.meshgrid(lon, lat) 
X,Y = m(x, y)

x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*20)
y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*30)

x2,y2 = np.meshgrid(x2,y2)
X2,Y2 = m(x2, y2)

data2 = interp(slope_map,x[0],y[:,0],x2,y2,order=1)

mdata = maskoceans(x2, y2, data2,resolution='h',grid=1.25,inlands=True)

print(mdata)

for i in xrange(1):
   fig = plt.figure(figsize=(48, 48)) 
   m.drawmapboundary(fill_color='white', zorder=-1)
   m.fillcontinents(color='0.8', lake_color='white', zorder=0)
 
   m.drawcoastlines(color='0.6', linewidth=0.5)
   m.drawcountries(color='0.6', linewidth=0.5)
   m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1],    dashes=[1,1], linewidth=0.25, color='0.5',fontsize='xx-large')
   m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')
 
   #PLOT ABSOLUTE
   #levels = np.linspace(0.,100.,21)
   #vmin,vmax = (0,100)
   #cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.jet,extend='both')
   
   #PLOT DIFFERENCE
   #vmin,vmax = (0.25,1.)
   vmin,vmax = (np.nanmean(slope_map)-np.nanstd(slope_map),np.nanmean(slope_map)+np.nanstd(slope_map))
   print(vmin,vmax)
   #vmin,vmax = (np.min(slope_map), np.max(slope_map))
   levels = np.linspace(vmin,vmax,16,endpoint=True)   
   #cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.RdYlGn_r,extend='both')
   cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.bwr,extend='both')
   
   plt.tight_layout()
   cbar = m.colorbar(cs,location='bottom',pad='10%',ticks=np.linspace(vmin,vmax,3),format='%.1f')
   cbar.ax.set_xticklabels(['Low risk','Medium risk','High risk'])
   #cbar.ax.text([0.0,0.5,1.0])
   #cbar.update_ticks()
   #cbar.ax.get_yaxis().labelpad = 60
   cbar.ax.get_xaxis().labelpad = 60
   #cbar.ax.set_ylabel('EM (%)', rotation=270)
   cbar.ax.set_xlabel('Nitrogen limitation', rotation=0,color='black', size=78)
   #cbar.ax.set_xlabel('ECM tree basal area (%)', rotation=0,color='black', size=78)
   cbar.solids.set_edgecolor("face")
   #cbar.set_clim(0.0,100)
   cbar.set_clim(vmin,vmax)
   #plt.title(r'Sulman et al. (2019) - Shi et al. (2016)')
   cbar.ax.tick_params(labelsize='xx-large')
   plt.savefig('offer_vs_demand_clm5_bwr_NPP_FRACTION.png',bbox_inches="tight")
   #plt.savefig('diff_em_tot_3.pdf',bbox_inches="tight",dpi=300)
   #plt.savefig('ecm_orig_shi_1p9x2p5.png',bbox_inches="tight",dpi=300)
   plt.show()

#sys.exit()
print(np.shape(average[:]))
y_av = smooth1(average[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
x_av = smooth1(time,120)
y_av_stein1 = smooth1(average_stein1[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av_stein2 = smooth1(average_stein2[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av_stein3 = smooth1(average_stein3[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

average1 = np.ma.average(nee_stein1,axis=1,weights=weights)
average1 = np.ma.average(average1,axis=1)

y_av1 = smooth1(averagek[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_stein1 = smooth1(averagek_stein1[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_stein2 = smooth1(averagek_stein2[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av1_stein3 = smooth1(averagek_stein3[:],12)*(1)*(365*24*60*60)*148847000*(1000*1000)/(10**15)

average2 = np.ma.average(nee_stein2,axis=1,weights=weights)
average2 = np.ma.average(average2,axis=1)

y_av2 = smooth1(averagei[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_stein1 = smooth1(averagei_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_stein2 = smooth1(averagei_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av2_stein3 = smooth1(averagei_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)


average3 = np.ma.average(nee_stein3,axis=1,weights=weights)
average3 = np.ma.average(average3,axis=1)

y_av3 = smooth1(averagej[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_stein1 = smooth1(averagej_stein1[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_stein2 = smooth1(averagej_stein2[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)
y_av3_stein3 = smooth1(averagej_stein3[:],12)*(1000)*(365*24*60*60)*148847000*(1000*1000)/(10**15)



y_av4_stein1 = smooth1(averagel_stein1[:],12)*100.







mean = np.mean(y_av[0:30*12*50])
mean1 = np.mean(y_av1[0:30*12*50])
mean2 = np.mean(y_av2[0:30*12*50])
mean3 = np.mean(y_av3[0:30*12*50])

print(np.shape(y_av1))
print(time)
start = 0
end = 30*12*161

xtick = np.arange(start, end,step = 12*25*30)

plt.style.use('ggplot')
SIZE = 12
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
for i in xrange(50):
   count = 38*i
   slope, intercept, r_value, p_value, std_err = stats.linregress(time[0+count:38+count],y_av[0+count:38+count])
   slope_av.append(slope)
   slope, intercept, r_value, p_value, std_err = stats.linregress(time[0+count:38+count],y_av1[0+count:38+count])
   slope_av1.append(slope)

slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av_stein1[:])
print('NPP_NUPTAKE',slope,100.*slope/np.mean(y_av_stein1[:]))
print('NPP_NUPTAKE','mean',np.mean(y_av_stein1[:]),'std',np.std(y_av_stein1[:]))

slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av1_stein1[:])
print('NPP',slope,100.*slope/np.mean(y_av1_stein1[:]))
print('NPP','mean',np.mean(y_av1_stein1[:]),'std',np.std(y_av1_stein1[:]))

slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av2_stein1[:])
print('NUPTAKE',slope,100.*slope/np.mean(y_av2_stein1[:]))
print('NUPTAKE','mean',np.mean(y_av2_stein1[:]),'std',np.std(y_av2_stein1[:]))

slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av3_stein1[:])
print('PLANT_DEMAND',slope,100.*slope/np.mean(y_av3_stein1[:]))
print('PLANT_DEMAND','mean',np.mean(y_av3_stein1[:]),'std',np.std(y_av3_stein1[:]))

slope, intercept, r_value, p_value, std_err = stats.linregress(time[:],y_av4_stein1[:])
print('NUPTAKE_NPP_FRACTION',slope,100.*slope/np.mean(y_av4_stein1[:]))
print('NUPTAKE_NPP_FRACTION','mean',np.mean(y_av4_stein1[:]),'std',np.std(y_av4_stein1[:]))


sys.exit()
#ax1.plot(time,y_av/y_av1,'k',label='NPP_NUPTAKE')
#ax1.plot(time,y_av_stein1/y_av1_stein1,'g',alpha=0.3,label='NPP_NUPTAKE')
#ax1.plot(time,y_av_stein2/y_av1_stein2,'b',alpha=0.3,label='NPP_NUPTAKE')
#ax1.plot(time,y_av_stein3/y_av1_stein3,'r',alpha=0.3,label='NPP_NUPTAKE')

ax1.plot(slope_av,'k',label='NPP_NUPTAKE')
ax1.plot(slope_av1,'r',label='NPP')
#ax1.plot(time,y_av2,'b',label='Steidinger et al. (2019)')
#ax1.plot(time,y_av3,'g',label='Soudzilovskaia et al. (2019)')
ax1.legend(loc='best',fontsize ="small",ncol=2,borderpad=0.)
#plt.ylim(0,16)
#ax1.set_ylim(-0.5,0.5)
#plt.xlim(start,end)
#ax1.set_xlim(start,end)
#plt.xticks(np.arange(start, end,step = 12*25*30), ('1850','1875','1900','1925','1950','1975','2000'))
#ax1.set_xticks(xtick)
ax1.set_xticklabels(['1850','1875','1900','1925','1950','1975','2000'])
#plt.ylabel(r'NCEM (TgN.yr$^{-1}$)')
ax1.set_ylabel(r'NPP (PgC.yr$^{-1}$)')
#plt.xlabel('year')
ax1.set_xlabel('year')

my_pal = {"k","r"}

vector = [slope_av,slope_av1]
#sns.boxplot(data = np.gradient(y_av),color='k',ax=ax2)
#sns.boxplot(data = np.gradient(y_av1),color='r',ax=ax2)
sns.boxplot(data = vector,palette=my_pal,ax=ax2)
#sns.boxplot(data = y_av2,color='b',ax=ax2)
#sns.boxplot(data = y_av3,color='g',ax=ax2)
#ax2.set_ylim(0,16)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_yticklabels('')
#ax2.set_yticks([])
ax2.set_xticklabels('')
#plt.title('US - UMB')
plt.savefig('NPP_gradient.png')
plt.show()


sys.exit()

perecm = params.variables['perecm'][:]
perecm_new = [0.7092, 0.7208, 0.7208, 0.7317, 0.6482, 0.6482, 0.6904, 0.6904,0.6904,0.7053, 0.7229, 0.7229, 0.7229, 0.7229, 0.7229,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601,0.6601, 0.6601]

perecm_new_2 = [0.7092, 0.7208, 0.7208, 0.7317, 0.6482, 0.6482, 0.6904, 0.6904,0.6904,0.7053, 0.7229, 0.7229, 0.7229, 0.7229, 0.7229,0.6601,0.6601,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0]

pft_map_nat = surf.variables['PCT_NAT_PFT'][:]
pft_map_crop = surf.variables['PCT_CFT'][:]

pct_natveg = surf.variables['PCT_NATVEG'][:]
pct_crop = surf.variables['PCT_CROP'][:]

#Declaring empty matrix
perecm_map_nat = np.zeros((96,144))
perecm_map_nat_new = np.zeros((96,144))

perecm_map_crop = np.zeros((96,144))
perecm_map_crop_new = np.zeros((96,144))

perecm_map_tot = np.zeros((96,144))
perecm_map_tot_new = np.zeros((96,144))

for i in xrange(14):
       #print(i,perecm[i],perecm_new[i])
       perecm_map_nat = perecm_map_nat + (perecm[i]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

       perecm_map_nat_new = perecm_map_nat_new + (perecm_new[i]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

perecm_map_nat = ma.array(perecm_map_nat,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_nat_new = ma.array(perecm_map_nat_new,mask=[pft_map_nat[0,:,:]>90.])

for i in xrange(64):
       #print(i+14,perecm[i+14],perecm_new[i+14])
       perecm_map_crop = perecm_map_crop + (perecm[i+14]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.
    
       perecm_map_crop_new = perecm_map_crop_new + (perecm_new[i+14]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.

perecm_map_crop = ma.array(perecm_map_crop,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_crop_new = ma.array(perecm_map_crop_new,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_tot = perecm_map_nat + perecm_map_crop
perecm_map_tot_new = perecm_map_nat_new + perecm_map_crop_new

lon = f.variables['lon'][:]
perecm_map_crop_new,lon = shiftgrid(180., perecm_map_crop_new, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_tot_new,lon = shiftgrid(180., perecm_map_tot_new, lon, start=False)

lon = f.variables['lon'][:]
nee,lon = shiftgrid(180., nee, lon, start=False)

lon = f.variables['lon'][:]
nee_stein,lon = shiftgrid(180., nee_stein, lon, start=False)

m = Basemap(projection='robin', lon_0=0.,resolution='l')

#x, y = m(lon, lat)
x,y = np.meshgrid(lon, lat) 
X,Y = m(x, y)

x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*10)
y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*10)

x2,y2 = np.meshgrid(x2,y2)
X2,Y2 = m(x2, y2)

con = []

for i in xrange(1932):
    iden = 100.*abs(nee_stein[i,:,:] - nee[i,:,:])/abs(nee[i,:,:])
    total = iden.count()
    iden = np.ma.masked_where(iden < 10., iden)
    non_masked = iden.count()
    perc = float(non_masked)/total
    con.append(100.*perc)
    print (i,max(con))
    plt.imshow(iden[::-1,:])
    plt.colorbar()
    plt.show()

plt.plot(con[:])
plt.show()
sys.exit()

for i in xrange(1):
      fig = plt.figure(figsize=(48, 48)) 
      m.drawmapboundary(fill_color='white', zorder=-1)
      m.fillcontinents(color='0.8', lake_color='white', zorder=0)
 
      m.drawcoastlines(color='0.6', linewidth=0.5)
      m.drawcountries(color='0.6', linewidth=0.5)
      m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1],    dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')
      m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')

      mask =  perecm_map_tot_new
      nee_stein = np.mean(nee_stein,axis=0)
      nee = np.mean(nee,axis=0)
      v = (nee_stein[:,:] - nee[:,:])*(365*24*60*60)
      v0 =  nee[:,:]*(365*24*60*60)
      diff = 100.*(v.mean()/v0.mean())
      maxi = 100.*(v.max()/v0.max())
      #print mapi,var,'mean',diff,'%','max',maxi

      mask_int = interp(mask,x[0],y[:,0],x2,y2,order=1)
      data2 = interp(v,x[0],y[:,0],x2,y2,order=1)
      mdata = maskoceans(x2, y2, data2,resolution='h',grid=1.25,inlands=True)
      #mdata = maskoceans(X, Y, v,resolution='l',grid=1.25,inlands=True)
   
      #print (v.max())
      #print (v.min())
     
      if abs(v.max()) < abs(v.min()):
         vmin,vmax = (-v.max(),v.max())
      else:
         vmin,vmax = (v.min(),-v.min())

      #print(long_name, unit)
      #print(vmin,vmax)

      if vmin == 0.0:
         break

      levels = np.linspace(vmin,vmax,13)
      #levels = np.linspace(-100.,100.,21)
      cs = m.contourf(X2,Y2,mdata*(mask_int/mask_int),levels,vmin=vmin,vmax=vmax,cmap=plt.cm.PuOr_r,extend='both')
      #cs = m.pcolormesh(X,Y,mdata*(v/v),cmap=plt.cm.Spectral_r)
      #cs = m.contourf(X,Y,mdata,levels,vmin=-100.0,vmax=100.0,cmap=plt.cm.bwr,extend='both')
   
      #cs = m.contourf(x,y,mdata)

      plt.tight_layout()
      if (vmax < 0.1):
         cbar = m.colorbar(cs,location='bottom',pad='10%',ticks=np.linspace(vmin,vmax,7),format='%.1e')
      else: 
         cbar = m.colorbar(cs,location='bottom',pad='10%',ticks=np.linspace(vmin,vmax,7),format='%.1f')
      #cbar.ax.get_yaxis().labelpad = 60
      cbar.ax.get_xaxis().labelpad = 60
      #cbar.ax.set_ylabel('EM (%)', rotation=270)
      cbar.ax.set_xlabel(r'Change in '+long_name+' ('+unit[0:2]+' m$^2$ yr$^{-1}$)', rotation=0,color='black', size=78)
      #cbar.solids.set_edgecolor("face")
      cbar.set_clim(vmin,vmax)
      cbar.ax.tick_params(labelsize='xx-large')
      #cbar.set_clim(-100,100)
      #plt.title(r'EM (%)  - PFT = BET')

      #plt.savefig('NEE_diff_2.pdf',bbox_inches="tight",dpi=300)
      plt.savefig('/home/renato/datasets/IHistClm50Bgc_outputs/gpp.png',bbox_inches="tight")
      #plt.show()
      #sys.exit()
      plt.close()
sys.exit()

