import math
import numpy as np
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

"""
This function calculates where a line intersects an ellipse, taking into account 
the rotation of the ellipse and the angle of the line. The steps require rotating
the line to the ellipse's coordinate system, setting up and solving 
a quadratic equation, and calculating the actual intersection points 
in the original coordinate system.
"""
# * * * * * * *
# Calculate the coordinates of the intersection point of an ellipse and a line
# * * * * * * *
def ellipse_line_intersection(Xc, Yc, L, W, phi, theta):
    # Xc, Yc: Center coordinates of the ellipse
    # L: Length of the major axis of the ellipse
    # W: Width of the minor axis of the ellipse
    # phi: Angle of rotation of the ellipse from the x-axis in radians
    # theta: Angle of the line relative to the x-axis in radians
    
    # Get semimajor and semiminor axes
    a = L / 2  # Calculate half-length of the major axis
    b = W / 2  # Calculate half-length of the minor axis

    # Calculate the parametric equations of the line
    dx = a * np.cos(theta)  # x-component of the direction vector of the line
    dy = a * np.sin(theta)  # y-component of the direction vector of the line
    
    # Rotate the line to align the ellipse with the coordinate axes
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x1_rot = (dx * cos_phi + dy * sin_phi)   # Rotate x-component of line
    y1_rot = (-dx * sin_phi + dy * cos_phi)  # Rotate y-component of line
    
    # Calculate oefficients of the quadratic equation At^2 + Bt + C = 0
    # t is a scalar multiplier to determine the distance along the direction vector of the line
    # from the center of the ellipse (Xc, Yc) to the intersection points with the ellipse.
    # Here we transform the line equation into ellipse's coordinate system
    # A is the coefficient of t^2
    A = (x1_rot / a) ** 2 + (y1_rot / b) ** 2  
    # B is the coefficient of t
    B = 2 * ((Xc * cos_phi + Yc * sin_phi - Xc * cos_phi - Yc * sin_phi) * x1_rot / a ** 2 + 
             (Xc * sin_phi - Yc * cos_phi - Xc * sin_phi + Yc * cos_phi) * y1_rot / b ** 2)  
    # Constant term
    C = ((Xc * cos_phi + Yc * sin_phi - Xc * cos_phi - Yc * sin_phi) ** 2 / a ** 2 +
         (Xc * sin_phi - Yc * cos_phi - Xc * sin_phi + Yc * cos_phi) ** 2 / b ** 2) - 1  
    
    # Solve the quadratic equation At^2 + Bt + C = 0
    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return []  # No intersection if discriminant is negative
    
    # Calculate solutions for t using the quadratic formula
    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    # Calculate the intersection points using the parametric line equation
    x1 = Xc + dx * t1  # x-coordinate of first intersection
    y1 = Yc + dy * t1  # y-coordinate of first intersection
    x2 = Xc + dx * t2  # x-coordinate of second intersection
    y2 = Yc + dy * t2  # y-coordinate of second intersection
    
    return [(x1, y1), (x2, y2)]  # Return both intersection points

# * * * * * * *
# Main program
# * * * * * * *

# Observed points obtained from digitized images
""" 
#  Example 1 * * * 
x1 = [86.3421, 8.6616, 0.0, 44.4931, 467.6329, 579.7773, 682.4396, 819.9304, 911.5605, 955.6889, 949.3979, 914.5693, 856.0354, 775.5285, 671.3163, 559.1719, 435.9955, 325.4922, 202.4069]
y1 = [813.2214, 701.1104, 566.1105, 445.3329, 71.111, 20.5555, 0.0, 30.0, 110.5554, 213.222, 330.1108, 418.5551, 525.8884, 625.4438, 713.8882, 789.5548, 844.7769, 868.4436, 865.2214]
"""

#  Example 2 * * * 
x1 = [578.8218, 667.6215, 734.6156, 791.4029, 752.1529, 715.7793, 650.2698, 592.0907, 513.4051, 436.2969, 360.5806, 284.7714, 203.0237, 500.0434, 415.6976, 332.3725, 258.048, 187.4351, 132.0397, 69.0355, 12.4338]
y1 = [910.3324, 878.2213, 824.3325, 752.9992, 483.6662, 375.9996, 276.8886, 198.3331, 134.2221, 92.111, 54.1111, 21.3333, 2.5556, 941.1102, 955.999, 947.4435, 925.9991, 884.4436, 833.9992, 779.777, 714.2215]

points = list(zip(x1,y1))
print(points)
a_points = np.array(points)
n1 = len(a_points)  # number of observed points
calc_x = [0]*n1     # coordinates of points along the elliptical curve
calc_y = [0]*n1
x = a_points[:, 0]  # coordinates of observed points
y = a_points[:, 1]
ell = EllipseModel()  # fit an ellipse to the observed points using a least squares method
ell.estimate(a_points)
#  retrieve the characteristics of the ellipse
xc, yc, a, b, phi = ell.params  # center(x,y), semimajor axis, semiminor axis, angle of rotation

print(f' Number of points = {n1}')
print(f' center (x,y): {xc:.3f}, {yc:.3f}')  # center of ellipse
print(f' semimajor axis = {a:,.4f}, semiminor axis = {b:,.4f}')
print(f' angle of rotation phi = {phi:.4f} ({np.rad2deg(phi):.3f} deg.)')

ddt = 0  # >0 to print debugging information

# Calculate error distance for each point
fitting_error = 0
n = 0
distance = 0
sum_of_error_distances = 0
while n < n1:
  x_o = x[n]  # observed x
  y_o = y[n]  # observed y
  # Calculate theta (angle to the observed point) = arctan( (y-yc)/(x-xc) )
  theta = math.atan2( (y_o - yc), (x_o - xc) )
  if theta < 0 :
    theta = math.radians(360) + theta    # correct for quadrants III and IV
  if ddt > 0 :
    print(f'x_o[{n}]={x_o:,.4f} y_o[{n}]={y_o:,.4f} theta={theta:.4f} ({np.rad2deg(theta):.3f} deg.)')

  # calculate predicted intersection point on elliptical curve based on theta
  intersection_points = ellipse_line_intersection(xc, yc, 2*a, 2*b, phi, theta)
  calc_x[n], calc_y[n] = intersection_points[0]  # select only the positive solution
  if ddt > 0: 
    print(f'   calc_x[{n}]={calc_x[n]:.3f}, calc_y[{n}]={calc_y[n]:.3f}')
  # distance between observed and predicted points
  distance = math.sqrt((x_o - calc_x[n])**2 + (y_o - calc_y[n])**2)  
  sum_of_error_distances += distance

  n = n + 1

average_error_distance = sum_of_error_distances/n
print(f' Average error distance = {average_error_distance:.4f}')
fitting_error = average_error_distance*100/b
print(f' Average error distance relative to semiminor axis = {fitting_error:.4f}%')

fig = plt.figure(figsize=(6, 6))
axs = plt.subplot()
axs.axis('equal')
axs.scatter(x, y, color='red')           # observed points are red dots
axs.plot(calc_x, calc_y, 'x')            # calculated points on ellipse are marked with x
axs.scatter(xc, yc, color='green', s=100)  # center is a green dot
axs.set_title(f' Number of points: {n1}\n Average error distance relative to semiminor axis: {fitting_error:.3f}%')
ell_patch = Ellipse(xy=(xc, yc), width=2*a, height=2*b, angle=phi*180/np.pi, edgecolor='b', facecolor='none')
axs.add_patch(ell_patch)
plt.show()
