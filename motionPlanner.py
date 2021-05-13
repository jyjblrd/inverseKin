import re
import numpy as np
from kinematics import Kinematics
from shortMath import sin, cos, rad
import shortMath
import quaternion
import math
import time

class MotionPlanner:

	def __init__(self):
		self.state = {
			"feed": "F100",
			"pos": {
				"xyz": np.array([120, 0, 100]),
				"q": np.quaternion(1, 0, 0, 0)
			}
		}

		self.queue = []

		self.segs_per_mm = 0.5 # number of interpolations per mm
		self.segs_per_rad = 45/math.pi # number of interpolations per radian

		self.kin = Kinematics()

	def file(self, file_name):
		'''
		Turns gcode file into joint angles

		Parameters
		  file_name: name of the gcode file
		'''

		start_time = time.time()

		file = open(file_name, "r")
		lines = file.readlines()
		lines = [x.strip() for x in lines] # remove whitespace characters

		for line in lines:
			self.gcode(line)

		print("Finished kinematic calculations in", time.time()-start_time, "seconds")

	def gcode(self, gcode):
		'''
		Take in GCode and put planned joint angles in queue

		Parameters
		  gcode: GCode command
		'''

		parsed_gcode = self.parse_gcode(gcode)

		if parsed_gcode["command"] in ["G0", "G1"]:
			self.move_command(parsed_gcode["parameters"])

	def move_command(self, parameters):
		
		v_0 = self.state["pos"]["xyz"].copy()
		q_0 = self.state["pos"]["q"].copy()

		# v_1 defaults to v_0 if no parameters given
		v_1 = self.state["pos"]["xyz"].copy()

		# q_1 defaults to q_0 if no parameters given
		yaw, pitch, roll = shortMath.quaternion_to_euler(self.state["pos"]["q"].copy())
		for parameter in parameters:
			if parameter[0] == "X":
				v_1[0] = float(parameter[1])
			elif parameter[0] == "Y":
				v_1[1] = float(parameter[1])
			elif parameter[0] == "Z":
				v_1[2] = float(parameter[1])
			elif parameter[0] == "RY":
				yaw = float(parameter[1])
			elif parameter[0] == "RP":
				pitch = float(parameter[1])
			elif parameter[0] == "RR":
				roll = float(parameter[1])
		q_1 = shortMath.euler_to_quaternion(yaw, pitch, roll)

		# if start position is same as end, exit
		if np.array_equal(v_0, v_1) and q_0 == q_1:
			return

		interpolated_positions = self.interpolate_move(v_0, v_1, q_0, q_1)

		for [v, q] in interpolated_positions:
			self.kin.calc_inv_kin(v, q)
			self.kin.calc_t_matrices()
			self.queue.append(self.kin.joint_angles.copy())

		# set state with new values
		self.state["pos"] = {
			"xyz": v_1,
			"q": q_1
		}

	def interpolate_move(self, v_0, v_1, q_0, q_1):
		'''
		Parameters
		  v_0: start vector
		  v_1: end vector
		  q_0: start quaternion
		  q_1: end quaternion

		Returns
		  List of interolated positions
		  [[v_0, q_0], [v_1, q_1],  [v_2, q_2], ...
		'''

		'''
		We want to determine how many interpolations we should do. We can
		interpolate by distance traveled or the angle of the end effector.
		We will compare these two methods and use the one with the greatest
		number of segments. 
		'''
		# find distance and angle between two positions
		dist = np.linalg.norm(v_0-v_1) # distance between two vectors
		theta = shortMath.angle_between_quaternions(q_0, q_1) # angle between two quaternions

		# find number of segments if done by distance or angle
		dist_segs = int(dist*self.segs_per_mm) # number of segments if determined by distance
		theta_segs = int(theta*self.segs_per_rad) # number of segments if determined by angle

		# use largest number of segments
		num_of_interpolations = max(dist_segs, theta_segs)

		# if dist = 0, we dont even need this normal vector (it is used to
		# interpolate the vector)
		if dist != 0:
			v_1_0_norm = (v_1-v_0)/dist # normal vector from v_0 to v_1
		else:
			v_1_0_norm = 0

		delta_dist = dist/num_of_interpolations
		
		interpolated_positions = [None]*num_of_interpolations

		for i in range(0, num_of_interpolations):
			interpolated_positions[i] = [
				v_0 + ((i+1)/num_of_interpolations) * dist * v_1_0_norm,
				quaternion.slerp(q_0, q_1, 0, 1, (i+1)/num_of_interpolations)
			]
		
		return interpolated_positions

	def parse_gcode(self, gcode):
		'''
		Turn GCode string "G1 X10 Y5" into {"command": "G1", parameters: [["X", 10], ["Y", 5]]}

		Parameters
		  gcode: GCode string to parse
		'''

		split_gcode = gcode.split(" ")

		parsed_gcode = {
			"command": split_gcode[0],
			"parameters": []
		}

		for parameter in split_gcode[1:]:
			parsed_gcode["parameters"].append(re.split(r'(^[^\d-]+)', parameter)[1:])

		return parsed_gcode #G0 X10 Y10 Z10