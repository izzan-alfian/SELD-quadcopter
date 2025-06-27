"""
A simple example of an animated plot... In 3D!
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
# This import is necessary for the '3d' projection to work
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def Gen_RandLine(length, dims=2) :
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length) :
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index-1] + step

    return lineData

def gen_rand_bar(length, columns):
    return np.array(np.random.rand(length, columns))

def update_lines(num, dataLines, lines, databars, bars) :
    for bar, data in zip(bars, databars) :
        ax_2d.clear()
        ax_2d.set_yticks(y_pos, labels=names)
        bar = ax_2d.barh(y_pos, data[num, :], align='center')

    for line, data in zip(lines, dataLines) :
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines

# Attaching 3D axis to the figure

fig = plt.figure(figsize=(10, 5))
ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
ax_3d.scatter(0.5, 0.5, 0.5)
ax_2d = fig.add_subplot(1, 2, 2)


# Fifty lines of random 3-D lines
data = [Gen_RandLine(25, 3) for index in range(50)]
values = [gen_rand_bar(25, 4) for index in range(1)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax_3d.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

names = ['group_a', 'group_b', 'group_c', 'group_d']
y_pos = np.arange(len(names))
ax_2d.set_yticks(y_pos, labels=names)
bars = [ax_2d.barh(y_pos, value[0, :], align='center') for value in values]
ax_2d.barh(y_pos, values[0, :], align='center')

# Setting the axes properties
ax_3d.set_xlim3d([0.0, 1.0])
ax_3d.set_xlabel('X')

ax_3d.set_ylim3d([0.0, 1.0])
ax_3d.set_ylabel('Y')

ax_3d.set_zlim3d([0.0, 1.0])
ax_3d.set_zlabel('Z')

ax_3d.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines, values, bars), interval=50, blit=True)

plt.tight_layout()
plt.show()