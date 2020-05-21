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

surf = Dataset('/home/renato/Steindinger/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190304.nc','r')
#surf = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c191015_steidinger_unmasked.nc','r')

surf_new = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c191015_steidinger_unmasked.nc','r')
surf_new_sou = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190821_soudzilovskaia.nc','r')
#surf_new_sul = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190821_sulman.nc','r')
surf_new_sul = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c200511_sulman.nc','r')

#surf_new = Dataset('/home/renato/datasets/surfacedata_clm/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr1850_c190304_steidinger_rcp85.nc','r')


params = Dataset('/home/renato/Steindinger/clm5_params.c171117.nc','r')
params_new = Dataset('/home/renato/datasets/sulman_2019/sulman_modified_clm5_params.c171117.nc','r')


#f1 = Dataset('/home/renato/src/shi_pft_ecms.nc','r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
clump = f.variables['GPP'][:]
clump1 = f1.variables['GPP'][:]
clump2 = f2.variables['GPP'][:]


perecm = params.variables['perecm'][:]
#perecm_new = params_new.variables['perecm'][:]
perecm_new = surf_new.variables['PERECM'][:]
perecm_new_sou = surf_new_sou.variables['PERECM'][:]
perecm_new_sul = surf_new_sul.variables['PERECM'][:]


pft_map_nat = surf.variables['PCT_NAT_PFT'][:]
pft_map_crop = surf.variables['PCT_CFT'][:]

pct_natveg = surf.variables['PCT_NATVEG'][:]
pct_crop = surf.variables['PCT_CROP'][:]

#Declaring empty matrix
perecm_map_nat = np.zeros((96,144))
perecm_map_nat_new = np.zeros((96,144))
perecm_map_nat_new_sou = np.zeros((96,144))
perecm_map_nat_new_sul = np.zeros((96,144))

perecm_map_crop = np.zeros((96,144))
perecm_map_crop_new = np.zeros((96,144))

perecm_map_tot = np.zeros((96,144))
perecm_map_tot_new = np.zeros((96,144))
perecm_map_tot_new_sou = np.zeros((96,144))
perecm_map_tot_new_sul = np.zeros((96,144))

for i in xrange(14):
    #print(i,perecm[i],perecm_new[i])
    perecm_map_nat = perecm_map_nat + (perecm[i]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

    #perecm_map_nat_new = perecm_map_nat_new + (perecm_new[i]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.
    perecm_map_nat_new = perecm_map_nat_new + (perecm_new[i,:,:]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

    perecm_map_nat_new_sou = perecm_map_nat_new_sou + (perecm_new_sou[i,:,:]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

    perecm_map_nat_new_sul = perecm_map_nat_new_sul + (perecm_new_sul[i,:,:]*pft_map_nat[i,:,:])*pct_natveg[:,:]/100.

perecm_map_nat = ma.array(perecm_map_nat,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_nat_new = ma.array(perecm_map_nat_new,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_nat_new_sou = ma.array(perecm_map_nat_new_sou,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_nat_new_sul = ma.array(perecm_map_nat_new_sul,mask=[pft_map_nat[0,:,:]>90.])


for i in xrange(64):
    #print(i+14,perecm[i+14],perecm_new[i+14])
    perecm_map_crop = perecm_map_crop + (perecm[i+14]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.
    
    #perecm_map_crop_new = perecm_map_crop_new + (perecm_new[i+14]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.
    perecm_map_crop_new = perecm_map_crop_new + (perecm_new[i+14,:,:]*pft_map_crop[i,:,:])*pct_crop[:,:]/100.

perecm_map_crop = ma.array(perecm_map_crop,mask=[pft_map_nat[0,:,:]>90.])

perecm_map_crop_new = ma.array(perecm_map_crop_new,mask=[pft_map_nat[0,:,:]>90.])

#perecm_map_tot = perecm_map_nat + perecm_map_crop
#perecm_map_tot_new = perecm_map_nat_new + perecm_map_crop_new

#ONLY NATURAL VEGETATION
perecm_map_tot = perecm_map_nat 
perecm_map_tot_new = perecm_map_nat_new 
perecm_map_tot_new_sou = perecm_map_nat_new_sou
perecm_map_tot_new_sul = perecm_map_nat_new_sul

#print lat,lon,clump
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

lon = f.variables['lon'][:]
perecm_map_tot_new_sou,lon = shiftgrid(180., perecm_map_tot_new_sou, lon, start=False)

lon = f.variables['lon'][:]
perecm_map_tot_new_sul,lon = shiftgrid(180., perecm_map_tot_new_sul, lon, start=False)

m = Basemap(projection='robin', lon_0=0.,resolution='l')


#x, y = m(lon, lat)
x,y = np.meshgrid(lon, lat) 
X,Y = m(x, y)

#print(lon,lat)
#print(len(lon),len(lat))



v0 = perecm_map_tot_new[:,:] - perecm_map_tot[:,:]
v1 = perecm_map_tot_new_sou[:,:] - perecm_map_tot[:,:]
v2 = perecm_map_tot_new_sul[:,:] - perecm_map_tot[:,:]
v3 = perecm_map_tot_new_sul[:,:] - perecm_map_tot_new[:,:]
v4 = perecm_map_tot_new_sou[:,:] - perecm_map_tot_new[:,:]
v5 = perecm_map_tot_new_sou[:,:] - perecm_map_tot_new_sul[:,:]
#v = perecm_map_tot_new[:,:]
#v = perecm_map_tot[:,:]

v = np.mean(np.array([v0,v1,v2,v3,v4,v5]),axis = 0)
vstd = np.std(np.array([v0,v1,v2,v3,v4,v5]),axis = 0)
vstd = np.std(np.array([v0,v1,v2]),axis = 0)


agree_matrix = np.zeros((96,144))


count_pos = 0
count_neg = 0

for i in xrange(96):
   for j in xrange(144):
      if (v0[i,j] > 0 and v1[i,j] > 0 and v2[i,j] > 0 ) or (v0[i,j] < 0 and v1[i,j] < 0 and v2[i,j] < 0 ) or (v0[i,j] == 0 and v1[i,j] == 0 and v2[i,j] == 0 ):
            agree_matrix[i,j] = (vstd[i,j]/np.max(vstd))
            count_pos = count_pos + 1
      else: 
            agree_matrix[i,j] = (-1.*vstd[i,j]/np.max(vstd))
            count_neg = count_neg + 1

      #agree_matrix[i,j] = vstd[i,j]
      #agree_matrix[i,j] = v[i,j]


print('mean=',np.nanmean(agree_matrix),'std=',np.nanstd(agree_matrix))
print('quart25=',np.nanquantile(agree_matrix,0.25),'quart75=',np.nanquantile(agree_matrix,0.75))

lon = f.variables['lon'][:]
pft_map_nat,lon = shiftgrid(180., pft_map_nat, lon, start=False)

agree_matrix = ma.array(agree_matrix,mask=[pft_map_nat[0,:,:]>90.])


#print('AGREE =',100.*count_pos/(count_pos+count_neg),'%')
#print('DISAGREE =',100.*count_neg/(count_pos+count_neg),'%')

agree = np.sum(np.ma.array(agree_matrix)>0)
neutral = np.sum(np.ma.array(agree_matrix)==0)
disagree = np.sum(np.ma.array(agree_matrix)<0)

print('AGREE =',100.*agree/(agree+disagree+neutral),'%')
print('NEUTRAL =',100.*neutral/(agree+disagree+neutral),'%')
print('DISAGREE =',100.*disagree/(agree+disagree+neutral),'%')
#plt.imshow(agree_matrix)
#plt.colorbar()
#plt.show()



x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*20)
y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*30)

#x2 = np.linspace(x[0][0],x[0][-1],x.shape[1]*2)
#y2 = np.linspace(y[0][0],y[-1][0],y.shape[0]*3)

x2,y2 = np.meshgrid(x2,y2)
X2,Y2 = m(x2, y2)

#data2 = interp(v,x[0],y[:,0],x2,y2,order=1)
data2 = interp(agree_matrix,x[0],y[:,0],x2,y2,order=1)


mdata = maskoceans(x2, y2, data2,resolution='h',grid=1.25,inlands=True)




for i in xrange(1):
   fig = plt.figure(figsize=(48, 48)) 
   m.drawmapboundary(fill_color='white', zorder=-1)
   m.fillcontinents(color='0.8', lake_color='white', zorder=0)
 
   m.drawcoastlines(color='0.6', linewidth=0.5)
   m.drawcountries(color='0.6', linewidth=0.5)
   m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1],    dashes=[1,1], linewidth=0.25, color='0.5',fontsize='xx-large')
   m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize='x-large')

 
   #PLOT ABSOLUTE
   levels = np.linspace(0.,50.,21)
   vmin,vmax = (0,50)
   cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.jet,extend='both')
   
   #PLOT DIFFERENCE
   #vmin,vmax = (0,100)
   vmin,vmax = (-1,1)
   levels = np.linspace(vmin,vmax,16)   
   #cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.RdYlGn,extend='both')
   cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.bwr_r,extend='both')
 
   #PLOT CHANGE
   #vmin,vmax = (-100,100)
   #levels = np.linspace(vmin,vmax,16)   
   #cs = m.contourf(X2,Y2,mdata,levels,vmin=vmin,vmax=vmax,cmap=plt.cm.bwr,extend='both')
   

   plt.tight_layout()
   cbar = m.colorbar(cs,location='bottom',pad='10%',ticks=np.linspace(vmin,vmax,3),format='%.1f')
   cbar.ax.set_xticklabels(['Low','Neutral','High'])
   #cbar.ax.get_yaxis().labelpad = 60
   cbar.ax.get_xaxis().labelpad = 60
   #cbar.ax.set_ylabel('EM (%)', rotation=270)
   #cbar.ax.set_xlabel('Standard deviation ECM tree basal area (%)', rotation=0,color='black', size=94)
   #cbar.ax.set_xlabel('Change in ECM tree basal area (%)', rotation=0,color='black', size=94)
   cbar.ax.set_xlabel('Agreement', rotation=0,color='black', size=94)
   cbar.solids.set_edgecolor("face")
   #cbar.set_clim(0.0,100)
   cbar.set_clim(vmin,vmax)
   #plt.title(r'Sulman et al. (2019) - Shi et al. (2016)')
   cbar.ax.tick_params(labelsize='xx-large')
   #plt.savefig('diff_myc_maps_larger_lr.png',bbox_inches="tight")
   plt.savefig('agreement_myc_maps_v2.png',bbox_inches="tight")
   #plt.savefig('diff_em_tot_3.pdf',bbox_inches="tight",dpi=300)
   #plt.savefig('ecm_orig_shi_1p9x2p5.png',bbox_inches="tight",dpi=300)
   #plt.show()

sys.exit()


