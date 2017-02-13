import numpy as np
import scipy as sc
import copy

sc = 11

map_d_217_uni = 2*np.random.randn(100,100)
map_l = 2*np.random.randn(100,100)
map_d_353_uni = map_d_217_uni*sc

map_217 = map_d_217_uni + map_l
map_353 = map_d_353_uni + map_l

# Corr coeff
R_uni = np.sum(map_217*map_353) / np.sqrt(np.sum(map_217**2) * np.sum(map_353**2))

# Now split dust up
map_d_217_nonuni = copy.deepcopy(map_d_217_uni)
map_d_217_nonuni[0:50,:]=map_d_217_nonuni[0:50,:]+map_d_217_nonuni[50:,:]
map_d_217_nonuni[50:,:]=0
map_d_353_nonuni = map_d_217_nonuni*sc

map_217b = map_d_217_nonuni + map_l
map_353b = map_d_353_nonuni + map_l

R_nonuni = np.sum(map_217b*map_353b) / np.sqrt(np.sum(map_217b**2) * np.sum(map_353b**2))



