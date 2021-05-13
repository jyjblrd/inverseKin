import math
import numpy as np
import quaternion

# make math functions shorter
def rad(degree):
	return math.radians(degree)

def cos(rad):
	return math.cos(rad)

def sin(rad):
	return math.sin(rad)

# multuple matrix multiplications
def mult_matmul(*arg):
	matrixs = [*arg]
	finalMatrix = matrixs[0]
	matrixs.pop(0)

	for matrix in matrixs:
		finalMatrix = np.matmul(finalMatrix, matrix)

	return finalMatrix

def euler_to_quaternion(yaw, pitch, roll):
	'''
	Convert euler angles to quaternion

	Parameters
	  yaw: yaw angle (rotation around x)
	  pitch: pitch angle (rotation around y)
	  roll: roll angle (rotation around z)
	'''

	qr = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
	qi = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
	qj = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
	qk = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)

	return np.quaternion(qr, qi, qj, qk)

def quaternion_to_euler(q):
	'''
	Convert quaternion to euler angles

	Parameters
	  q: Quaternion object
	'''

	(x, y, z, w) = (q.x, q.y, q.z, q.w)
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll = math.atan2(t0, t1)
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch = math.asin(t2)
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw = math.atan2(t3, t4)

	return yaw, pitch, roll

def angle_between_quaternions(q_0, q_1):
	'''
	Find angle between two quaternions

	Parameters
	  q_0: first quaternion
	  q_1: second quaternion

	Result
	  Theta
	'''

	q_0_matrix = np.array([q_0.w, q_0.x, q_0.y, q_0.z])
	q_1_matrix = np.array([q_1.w, q_1.x, q_1.y, q_1.z])

	return math.acos(2*np.dot(q_0_matrix, q_1_matrix)**2-1)