import numpy
import tests.visualization as visualization
robot_radius = 0.25
bounds = numpy.array([0, 5, -3, 3])
num_points = numpy.array([101, 101])
polygon_list = []
xy = numpy.array([[2.518,1.83,2.043,2.406,2.655,2.518], [0.5048,0.2963,-0.2348,-0.8039,-0.0533,0.5048]]).transpose()
polygon_list.append(xy)
xy1 = numpy.array([[2.518,1.83,2.043,2.406,2.655,2.518], [0.5048,0.2963,-0.2348,-0.8039,-0.0533,0.5048]]).transpose()
polygon_list.append(xy1)
xy2 = numpy.array([[0,5,5,0,0,4,4,0,0], [0,0,5,5,4,4,1,1,0]]).transpose()
polygon_list.append(xy2)
diffeo_params = dict()
diffeo_params['p'] = 20
diffeo_params['epsilon'] = 1.5
diffeo_params['varepsilon'] = 1.5
diffeo_params['mu_1'] = 0.5
diffeo_params['mu_2'] = 0.01
diffeo_params['workspace'] = numpy.array([[-100,-100],[100,-100],[100,100],[-100,100],[-100,-100]])
visualization.visualize_diffeoDeterminant_triangulation(polygon_list, robot_radius, bounds, num_points, diffeo_params)