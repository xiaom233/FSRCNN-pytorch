import numpy
import h5py
f = h5py.File('BLAH_BLAH/91-res.h5', 'r') #91-image_x3++  91-res
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    print(f[key][0])
    print('```````````````````````````````')
    print(numpy.squeeze(f[key][0]))