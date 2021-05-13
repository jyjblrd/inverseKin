import scipy
from scipy import interpolate
import numpy as np

#This is your data, but we're 'zooming' into just 5 data points
#because it'll provide a better visually illustration
#also we need to transpose to get the data in the right format
data = np.array([
    [1.926,1.744,2755.868],
    [7.014,16.351,2854.041],
    [7.274,30.83,2951.83],
    [2.685,40.163,3049.031],
]).transpose()

#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data)
#here we generate the new interpolated dataset, 
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,500), tck)

#now lets plot it!
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(data[0], data[1], data[2], label='originalpoints', lw =2, c='Dodgerblue')
ax.plot(new[0], new[1], new[2], label='fit', lw =2, c='red')
ax.legend()
plt.show()