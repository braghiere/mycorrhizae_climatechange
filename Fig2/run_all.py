import os 

print('Plotting changes in ECM fraction between all maps...')
os.system('python /home/renato/datasets/figures/fig_larger/Fig2/plot_world_change_ecm.py')
print('Done!')

print('Plotting STD in ECM fraction between all maps...')
os.system('python /home/renato/datasets/figures/fig_larger/Fig2/plot_world_std.py')
print('Done!')

print('Plotting world agreement in ECM fraction between all maps...')
os.system('python /home/renato/datasets/figures/fig_larger/Fig2/plot_world_agreement.py')
print('Done!')
