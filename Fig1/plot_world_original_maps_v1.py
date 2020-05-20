from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap, cm, shiftgrid
from netCDF4 import Dataset, date2index
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import numpy.ma as ma
import numpy as np
from mpl_toolkits.basemap import maskoceans
from mpl_toolkits.basemap import interp
from matplotlib.colors import ListedColormap
import seaborn as sns
# High resolution cost lines
#http://introtopython.org/visualization_earthquakes.html
# High resolution cost lines
#http://basemaptutorial.readthedocs.io/en/latest/utilities.html
plt.style.use('ggplot')
# INCREASE SIZE IN 20%
#SIZE = 48
SIZE = 58
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)  # # size of the figure title

#f = Dataset('clumping005.nc','r')
f = Dataset('/home/renato/Desktop/clm5.0_NOFUN.clm2.h0.time.0102-1012.nc','r')
f1 = Dataset('/home/renato/Desktop/clm5.0_FUN.clm2.h0.time.0102-1012.nc','r')
f2 = Dataset('/home/renato/Desktop/clm5.0_FUN_NEWPERECM.clm2.h0.time.0102-1012.nc','r')

#Shi et al. 2016
surf = Dataset('/home/renato/Steindinger/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190304.nc','r')
params = Dataset('/home/renato/Steindinger/clm5_params.c171117.nc','r')

#Steidinger et al. 2019
#surf_ori = Dataset('/home/renato/datasets/shi_2016/steidinger_ecm_pft_1p9x2p5_reg_unmasked.nc','r')
surf_ori = Dataset('/home/renato/Steindinger/SupplementalFiles_Steidinger_etal2019/em_1degree.nc','r')
surf_new = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c191015_steidinger_unmasked.nc','r')
params_new = Dataset('/home/renato/datasets/steidinger_2019/1p9x2p5/steidinger_modified_clm5_params.c171117.nc','r')

#Sulman et al. 2019
#surf_ori = Dataset('/home/renato/datasets/shi_2016/soudzi_ecm_pft_1p9x2p5_reg.nc','r')
#surf_new = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190821_sulman.nc','r')
#params_new = Dataset('/home/renato/datasets/sulman_2019/sulman_modified_clm5_params.c171117.nc','r')

#Soudzilovskaia et al. 2019
#surf_new = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190821_soudzilovskaia.nc','r')
#params_new = Dataset('/home/renato/datasets/soudzilovskaia_2019/soudzilovskaia_modified_clm5_params.c171117.nc','r')

#f1 = Dataset('/home/renato/src/shi_pft_ecms.nc','r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]

perecm = params.variables['perecm'][:]
perecm_new = surf_new.variables['PERECM'][:]
#Original
#perecm_ori = surf_ori.variables['perecm (fraction)'][:]
#Steidinger
perecm_ori = surf_ori.variables['layer'][:]

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
    print(i,perecm[i],perecm_new[i])
    perecm_map_nat = perecm_map_nat + (perecm[i]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

    #perecm_map_nat_new = perecm_map_nat_new + (perecm_new[i]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.
    perecm_map_nat_new = perecm_map_nat_new + (perecm_new[i,:,:]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

perecm_map_nat = ma.array(perecm_map_nat,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_nat_new = ma.array(perecm_map_nat_new,mask=[pft_map_nat[0,:,:]>90.])



for i in xrange(64):
    print(i+14,perecm[i+14],perecm_new[i+14])
    perecm_map_crop = perecm_map_crop + (perecm[i+14]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.
    
    #perecm_map_crop_new = perecm_map_crop_new + (perecm_new[i+14]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.
    perecm_map_crop_new = perecm_map_crop_new + (perecm_new[i+14,:,:]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.

perecm_map_crop = ma.array(perecm_map_crop,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_crop_new = ma.array(perecm_map_crop_new,mask=[pft_map_nat[0,:,:]>90.])

# NATURAL + CROP VEGETATION
perecm_map_tot = perecm_map_nat + perecm_map_crop
perecm_map_tot_new = perecm_map_nat_new + perecm_map_crop_new

#ONLY NATURAL VEGETATION
#perecm_map_tot = perecm_map_nat 
#perecm_map_tot_new = perecm_map_nat_new 


#lon = np.clip(lon, -180., 180.)
#lon, lat = np.meshgrid(np.linspace(-180, 180, 144), np.linspace(-90, 90, 96))

# shifting grid to run from -180 to 180 rather than 0-360
#clump,lon = shiftgrid(180., clump, lon, start=False)
perecm_map_nat,lon = shiftgrid(180., perecm_map_nat, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_nat_new,lon = shiftgrid(180., perecm_map_nat_new, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_tot,lon = shiftgrid(180., perecm_map_tot, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_crop,lon = shiftgrid(180., perecm_map_crop, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_crop_new,lon = shiftgrid(180., perecm_map_crop_new, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_tot_new,lon = shiftgrid(180., perecm_map_tot_new, lon, start=False)


m = Basemap(projection='robin', lon_0=0.,resolution='l')

lat = surf_ori.variables['latitude'][:]
lat = np.arange(-89.5,90.0,1.0)
lon = surf_ori.variables['longitude'][:]

v = 100.*perecm_ori[:,:]


#x, y = m(lon, lat)
x,y = np.meshgrid(lon, lat) 
X,Y = m(x, y)

print(lon,lat)
print(len(lon),len(lat))


perecm_ori_plus = np.empty((180,360))

for i in xrange(180):
    if i < 5:
        perecm_ori_plus[i,:] = perecm_ori[145,:]
    if 5 < i < 151:
        perecm_ori_plus[i,:] = perecm_ori[i-5,:]  
    else:
        perecm_ori_plus[i,:] = perecm_ori[145,:]


v = 100.*perecm_ori_plus[:,:]


#v = perecm_map_tot_new[:,:] - perecm_map_tot[:,:]
#v = perecm_map_tot_new[:,:]
#v = 100.*perecm_ori[:,:]

x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*50)
y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*50)

#x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*40)
#y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*60)

x2,y2 = np.meshgrid(x2,y2)
X2,Y2 = m(x2, y2)

data2 = interp(v,x[0],y[:,0],x2,np.flipud(y2),order=1)

#print x2,y2


mdata = maskoceans(x2, y2, data2,resolution='h',grid=1.25,inlands=True)
#mdata = maskoceans(x2, y2, data2)

mdata[mdata<0] = np.nan

for i in xrange(1):
   fig = plt.figure(figsize=(48, 48)) 
   m.drawmapboundary(fill_color='white', zorder=-1)
   m.fillcontinents(color='0.8', lake_color='white', zorder=0)
 
   m.drawcoastlines(color='0.6', linewidth=0.5)
   m.drawcountries(color='0.6', linewidth=0.5)
   m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1],    dashes=[1,1], linewidth=0.25, color='0.5',fontsize='xx-large')
   m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')

 
   #PLOT ABSOLUTE
   levels = np.linspace(0.,100.,21)
   vmin,vmax = (0,100)
   cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.jet,extend='both')
   
   #PLOT DIFFERENCE
   #vmin,vmax = (-100,100)
   #levels = np.linspace(vmin,vmax,16)   
   #cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.bwr,extend='both')
   

   plt.tight_layout()
   cbar = m.colorbar(cs,location='bottom',pad='10%',ticks=np.linspace(vmin,vmax,7),format='%.1f')
   #cbar.ax.get_yaxis().labelpad = 60
   cbar.ax.get_xaxis().labelpad = 60
   #cbar.ax.set_ylabel('EM (%)', rotation=270)
   #cbar.ax.set_xlabel('Change in ECM tree basal area (%)', rotation=0,color='black', size=78)

   # INCREASE FONT SIZE IN 20%
   #cbar.ax.set_xlabel('ECM tree basal area (%)', rotation=0,color='black', size=78)
   cbar.ax.set_xlabel('Fraction of ECM (%)', rotation=0,color='black', size=94)
   cbar.solids.set_edgecolor("face")
   #cbar.set_clim(0.0,100)
   cbar.set_clim(vmin,vmax)
   #plt.title(r'Sulman et al. (2019) - Shi et al. (2016)')
   cbar.ax.tick_params(labelsize='xx-large')
   #plt.savefig('em_steindinger_tot.pdf',bbox_inches="tight",dpi=300)
   #plt.savefig('diff_em_tot_3.pdf',bbox_inches="tight",dpi=300)
   plt.savefig('ecm_orig_steidinger_1p0x1p0_larger_v2.png',bbox_inches="tight")
   plt.show()


sys.exit()


