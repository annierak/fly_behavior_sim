''' This script looks at conc/puff objects to check
 if they finished generating without error '''

import numpy as np
import sys
from pompy import data_importers


conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.
release_delay = 20*60.

importedPlumes = data_importers.ImportedPlumes(plume_file,
        array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,
        box_approx=True,epsilon = 0.0001)

# importedConc = data_importers.ImportedConc(conc_file,release_delay)

time = 50*60.
time = 2405.25

# for t in np.arange(0,time,0.25):
#     print(t)
try:
    test_p = importedPlumes.value(time,np.array([0]),np.array([0]))
    # test_p = importedPlumes.value(t,3000*np.random.randn(20),
    # 3000*np.random.randn(20))
except(IOError):
    print('Plume file not complete')

# try:
#     test_c = importedConc.value(time,np.array([0]),np.array([0]))
# except:
#     print('Conc file not complete')
