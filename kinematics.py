import numpy as np
from shortMath import sin, cos, rad, mult_matmul
import math
import quaternion

class Kinematics:

	def __init__(self):
		self.joint_angles = np.array([rad(x) for x in [0, -90, 0, 0, 0, 0]])
		'''
		Denavit-Hartenberg Parameters
		Col: t a d r
		(all compared to axies of next frame)
		t: angle around z axies (rad)
		a: angle around x axies (rad)
		d: distance along z axis (mm)
		r: distance along x axies (mm)
		https://automaticaddison.com/how-to-find-denavit-hartenberg-parameter-tables/
		'''

		self.dh_params = np.array([
			[self.joint_angles[0], rad(-90), 0,   10 ],
			[self.joint_angles[1], 0,        0,   100],
			[self.joint_angles[2], rad(-90), 0,   0  ],
			[self.joint_angles[3], rad(90),  100, 0  ],
			[self.joint_angles[4], rad(-90), 0,   0  ],
			[self.joint_angles[5], 0,        10,  0  ]
		])

		# transformation matricies
		self.t_matrix_0_1 = TMatrix() # trans. matrix from frame0 to frame1
		self.t_matrix_1_2 = TMatrix() # etc.
		self.t_matrix_2_3 = TMatrix()
		self.t_matrix_3_4 = TMatrix()
		self.t_matrix_4_5 = TMatrix()
		self.t_matrix_5_6 = TMatrix()

		self.t_matrix_0_2 = TMatrix()
		self.t_matrix_0_3 = TMatrix()
		self.t_matrix_0_4 = TMatrix()
		self.t_matrix_0_5 = TMatrix()
		self.t_matrix_0_6 = TMatrix()

		self.calc_t_matrices()

	def dh_param_to_t_matrix(self, dh_params):
		'''
		Convert DH parameters into a transformation matrix
		'''

		t, a, d, r = dh_params
		return np.array([
			[cos(t), -sin(t)*cos(a), sin(t)*sin(a),  r*cos(t)],
			[sin(t), cos(t)*cos(a),  -cos(t)*sin(a), r*sin(t)],
			[0,      sin(a),         cos(a),         d       ],
			[0,      0,              0,              1       ]
		])

	def calc_t_matrices(self):
		'''
		Calculate transformation matricies for all joints
		'''

		self.t_matrix_0_1.set_mat(self.dh_param_to_t_matrix(self.dh_params[0]))
		self.t_matrix_1_2.set_mat(self.dh_param_to_t_matrix(self.dh_params[1]))
		self.t_matrix_2_3.set_mat(self.dh_param_to_t_matrix(self.dh_params[2]))
		self.t_matrix_3_4.set_mat(self.dh_param_to_t_matrix(self.dh_params[3]))
		self.t_matrix_4_5.set_mat(self.dh_param_to_t_matrix(self.dh_params[4]))
		self.t_matrix_5_6.set_mat(self.dh_param_to_t_matrix(self.dh_params[5]))

		self.t_matrix_0_2.set_mat(np.matmul(self.t_matrix_0_1.mat,  self.t_matrix_1_2.mat))
		self.t_matrix_0_3.set_mat(np.matmul(self.t_matrix_0_2.mat,  self.t_matrix_2_3.mat))
		self.t_matrix_0_4.set_mat(np.matmul(self.t_matrix_0_3.mat,  self.t_matrix_3_4.mat))
		self.t_matrix_0_5.set_mat(np.matmul(self.t_matrix_0_4.mat,  self.t_matrix_4_5.mat))
		self.t_matrix_0_6.set_mat(np.matmul(self.t_matrix_0_5.mat,  self.t_matrix_5_6.mat))

	def update_joint_angles(self, joint_angles, **kwargs):
		'''
		Change the joint angles

		Parameters
		  joint_angles: numpy array with 6 angles (leave angle as None if you
		    dont want to update it)
		  is_rad: are the angles in radians (default: True)
		'''

		# change if radians if not already radians
		if "is_rad" in kwargs.keys() and kwargs["is_rad"] is False:
			joint_angles = [math.radians(x) for x in joint_angles]

		# only change angle if not None
		for i in range(0, 6):
			if joint_angles[i] is not None:
				self.joint_angles[i] = joint_angles[i]

		self.dh_params[:,0] = np.matrix([
			self.joint_angles[0],
			self.joint_angles[1],
			self.joint_angles[2],
			self.joint_angles[3],
			self.joint_angles[4],
			self.joint_angles[5]
		])

	def calc_inv_kin(self, pos, q):
		'''
		Calculate joint angles so that end effector reaches position inputted
		and is at the inputted angle

		Parameters
		  pos: vector representing position that you want the wrist to get to 
		  q: quaternion represeting angle of end effector
		'''

		# create r_matrix from quaternion
		r_matrix = quaternion.as_rotation_matrix(q)

		new_joint_angles = np.array([0, 0, 0])

		# find vector representing joint 5 position
		v_5_6 = np.matmul(r_matrix, np.matrix([[self.dh_params[5][2]],[0],[0]])) # vector from joint 5 to joint 6
		v_5_6 = np.ravel(v_5_6)
		pos_5 = np.subtract(pos, v_5_6)

		# find joint 1 theta
		t1 = math.atan2(pos_5[1], pos_5[0])

		# find joint 2 position
		d1 = self.dh_params[0,2]
		r1 = self.dh_params[0,3]
		j2 = np.array([r1*cos(t1), r1*sin(t1), d1])

		# find side lengths of SSS triangle
		d4 = self.dh_params[3,2]
		r2 = self.dh_params[1,3]
		a = np.linalg.norm(pos_5-j2)

		# find joint 2 theta
		t2 = math.acos((a**2+r2**2-d4**2) / (2*a*r2)) + math.atan2(pos_5[2]-d1, math.sqrt(pos_5[0]**2+pos_5[1]**2)-r1)
		t2 *= -1

		# find joint 3 theta
		t3 = math.acos((d4**2+r2**2-a**2) / (2*d4*r2))
		t3 *= -1
		t3 += rad(90)

		# calculate first 3 joints so we can solve t4 t5 and t6
		self.update_joint_angles(np.array([t1, t2, t3, None, None, None]), is_rad=True)
		self.calc_t_matrices()

		'''
		solve for t4 t5 and t6
		1. find equation for symbolic r_3_6 which is r_3_4*r_4_5*r_5_6
		2. use symbolic r_3_6 to figure out how to turn r_matrix into t4, t5 and t6
		see working on iPad
		'''
		y, p, r = math.pi, -math.pi/2, 0
		r_corr = np.matrix([
			[cos(y)*cos(p), cos(y)*sin(p)*sin(r)-sin(y)*cos(r), cos(y)*sin(p)*cos(r)+sin(y)*sin(r)],
			[sin(y)*cos(p), sin(y)*sin(p)*sin(r)+cos(y)*cos(r), sin(y)*sin(p)*cos(r)-cos(y)*sin(r)],
			[-sin(p),       cos(p)*sin(r),                      cos(p)*cos(r)                     ]
		])
		r_matrix_3_6 = mult_matmul(np.transpose(self.t_matrix_0_3.rot), r_matrix, np.transpose(r_corr))


		t4 = math.atan2(r_matrix_3_6[1,2], r_matrix_3_6[0,2])

		t5 = math.atan2(math.sqrt(r_matrix_3_6[0,2]**2+r_matrix_3_6[1,2]**2), r_matrix_3_6[2,2])
		t5 *= -1

		t6 = math.atan2(-r_matrix_3_6[2,1], r_matrix_3_6[2,0])
		t6 += rad(180)
		
		self.update_joint_angles(np.array([None, None, None, t4, t5, t6]), is_rad=True)

class TMatrix:

	def __init__(self, matrix=None):
		if matrix is not None:
			self.mat = matrix
			self.pos = matrix[0:3,3]
			self.rot = matrix[:3,:3]

	def set_mat(self, matrix):
		'''
		Set the transformation matrix

		Parameters:
		  matrix: transformation matrix
		'''

		self.mat = matrix
		self.pos = matrix[0:3,3]
		self.rot = matrix[:3,:3]