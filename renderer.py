import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import math
import time
import quaternion

from kinematics import Kinematics, TMatrix
from shortMath import cos, sin, euler_to_quaternion
from motionPlanner import MotionPlanner

class Renderer:

	def __init__(self, kin, mot):
		self.kin = kin
		self.mot = mot

		# mathplot lib stuff
		self.fig = plt.figure(figsize=(12, 10), dpi=80)
		self.ax = self.fig.gca(projection='3d')

	def draw_line(self, start_vector, end_vector, color="black"):
		'''
		Draw line from start_vector to end_vector

		Parameters
		  start_vector: start point of the line
		  end_vector: end point of the line
		  color: color of the line
		'''
		self.ax.plot3D(*zip(start_vector, end_vector), color=color)

	def draw_axies(self, origin, r_matrix):
		'''
		Draw XYZ axies lines

		Parameters
		  origin: vector representing origin of axies
		  r_matrix: rotation matrix representing how axies should be rotated
		'''

		x_axis = np.ravel(np.matmul(r_matrix, np.matrix([[10],[0],[0]]))) + origin
		y_axis = np.ravel(np.matmul(r_matrix, np.matrix([[0],[10],[0]]))) + origin
		z_axis = np.ravel(np.matmul(r_matrix, np.matrix([[0],[0],[10]]))) + origin
		self.draw_line(origin, x_axis, "red")
		self.draw_line(origin, y_axis, "green")
		self.draw_line(origin, z_axis, "blue")


	def set_axes_equal(self):
	    '''
	    Make axes of 3D plot have equal scale so that spheres appear as spheres,
	    cubes as cubes, etc..  This is one possible solution to Matplotlib's
	    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	    Input
	      ax: a matplotlib axis, e.g., as output from plt.gca().
	    '''

	    x_limits = self.ax.get_xlim3d()
	    y_limits = self.ax.get_ylim3d()
	    z_limits = self.ax.get_zlim3d()

	    x_range = abs(x_limits[1] - x_limits[0])
	    x_middle = np.mean(x_limits)
	    y_range = abs(y_limits[1] - y_limits[0])
	    y_middle = np.mean(y_limits)
	    z_range = abs(z_limits[1] - z_limits[0])
	    z_middle = np.mean(z_limits)

	    # The plot bounding box is a sphere in the sense of the infinity
	    # norm, hence I call half the max range the plot radius.
	    plot_radius = 0.5*max([x_range, y_range, z_range])

	    self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	    self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	    self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

	def draw_joint_sliders(self):
		'''
		Draw all of the joint angle sliders
		'''

		joint1_t = plt.axes([0.25, 0.1, 0.65, 0.015])
		self.joint1_t_slider = Slider(
		    ax=joint1_t,
		    label="Joint 1",
		    valmin=-180,
		    valmax=180,
		    valinit=self.kin.joint_angles[0],
		)
		joint2_t = plt.axes([0.25, 0.08, 0.65, 0.015])
		self.joint2_t_slider = Slider(
		    ax=joint2_t,
		    label="Joint 2",
		    valmin=-180,
		    valmax=180,
		    valinit=self.kin.joint_angles[1],
		)
		joint3_t = plt.axes([0.25, 0.06, 0.65, 0.015])
		self.joint3_t_slider = Slider(
		    ax=joint3_t,
		    label="Joint 3",
		    valmin=-180,
		    valmax=180,
		    valinit=self.kin.joint_angles[2],
		)
		joint4_t = plt.axes([0.25, 0.04, 0.65, 0.015])
		self.joint4_t_slider = Slider(
		    ax=joint4_t,
		    label="Joint 4",
		    valmin=-180,
		    valmax=180,
		    valinit=self.kin.joint_angles[3],
		)
		joint5_t = plt.axes([0.25, 0.02, 0.65, 0.015])
		self.joint5_t_slider = Slider(
		    ax=joint5_t,
		    label="Joint 5",
		    valmin=-180,
		    valmax=180,
		    valinit=self.kin.joint_angles[4],
		)
		joint6_t = plt.axes([0.25, 0, 0.65, 0.015])
		self.joint6_t_slider = Slider(
		    ax=joint6_t,
		    label="Joint 6",
		    valmin=-180,
		    valmax=180,
		    valinit=self.kin.joint_angles[5],
		)

		def joint_slider_update(val):
			'''
			Function called when joint angle slider changes
			'''

			self.kin.update_joint_angles(np.array([
				self.joint1_t_slider.val,
				self.joint2_t_slider.val,
				self.joint3_t_slider.val,
				self.joint4_t_slider.val,
				self.joint5_t_slider.val,
				self.joint6_t_slider.val,
			]), is_rad=False)
			self.kin.calc_t_matrices()
			
			self.clear()
			self.draw_arm(self.kin)
			self.render()

			print([math.degrees(x) for x in self.kin.joint_angles])
			print(self.kin.t_matrix_0_5.pos)
			print("*********")

		self.joint1_t_slider.on_changed(joint_slider_update)
		self.joint2_t_slider.on_changed(joint_slider_update)
		self.joint3_t_slider.on_changed(joint_slider_update)
		self.joint4_t_slider.on_changed(joint_slider_update)
		self.joint5_t_slider.on_changed(joint_slider_update)
		self.joint6_t_slider.on_changed(joint_slider_update)

	def draw_pos_sliders(self):
		'''
		Draw sliders which can change the wrist location (joint 5)
		'''

		x = plt.axes([0.01, 0.25, 0.015, 0.65])
		self.x_slider = Slider(
		    ax=x,
		    label="X",
		    valmin=-200,
		    valmax=200,
		    valinit=self.kin.t_matrix_0_6.pos[0],
		    orientation="vertical"
		)
		y = plt.axes([0.04, 0.25, 0.015, 0.65])
		self.y_slider = Slider(
		    ax=y,
		    label="Y",
		    valmin=-200,
		    valmax=200,
		    valinit=self.kin.t_matrix_0_6.pos[1],
		    orientation="vertical"
		)
		z = plt.axes([0.07, 0.25, 0.015, 0.65])
		self.z_slider = Slider(
		    ax=z,
		    label="Z",
		    valmin=-200,
		    valmax=200,
		    valinit=self.kin.t_matrix_0_6.pos[2],
		    orientation="vertical"
		)
		yaw = plt.axes([0.1, 0.25, 0.015, 0.65])
		self.yaw_slider = Slider(
		    ax=yaw,
		    label="Yaw",
		    valmin=-180,
		    valmax=180,
		    valinit=0,
		    orientation="vertical"
		)
		pitch = plt.axes([0.13, 0.25, 0.015, 0.65])
		self.pitch_slider = Slider(
		    ax=pitch,
		    label="Pitch",
		    valmin=-180,
		    valmax=180,
		    valinit=0,
		    orientation="vertical"
		)
		roll = plt.axes([0.16, 0.25, 0.015, 0.65])
		self.roll_slider = Slider(
		    ax=roll,
		    label="Roll",
		    valmin=-180,
		    valmax=180,
		    valinit=0,
		    orientation="vertical"
		)

		def pos_slider_update(val):
			'''
			Function called when position slider changes
			'''

			# rotation matrix where we want end effector
			yaw = math.radians(self.yaw_slider.val)
			pitch = math.radians(self.pitch_slider.val)
			roll = math.radians(self.roll_slider.val)

			'''
			# rotation matrix from euler angles (we'll swtich to quaternion later)
			r_matrix_0_6 = np.matrix([
				[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
				[sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
				[-sin(pitch),       cos(pitch)*sin(roll),                      cos(pitch)*cos(roll)                     ]
			])
			'''
			q = euler_to_quaternion(yaw, pitch, roll)

			self.kin.calc_inv_kin(
				np.array([
					self.x_slider.val,
					self.y_slider.val,
					self.z_slider.val
				]),
				q
			)
			self.kin.calc_t_matrices()

			self.clear()
			self.draw_arm(self.kin)
			self.render()

			print([round(x, 3) for x in self.kin.joint_angles])
			print(self.kin.t_matrix_0_5.pos)
			print("*********")

		self.x_slider.on_changed(pos_slider_update)
		self.y_slider.on_changed(pos_slider_update)
		self.z_slider.on_changed(pos_slider_update)
		self.yaw_slider.on_changed(pos_slider_update)
		self.pitch_slider.on_changed(pos_slider_update)
		self.roll_slider.on_changed(pos_slider_update)

	def draw_text_box(self):
		axLabel = plt.axes([0.25, 0.95, 0.30, 0.03])
		self.gcode_textbox = TextBox(axLabel, 'GCode: ', initial="G0 X120 Y0 Z10 RY1.57 RP0 RR0")

		def submit(text):
			self.mot.gcode(text)
			self.run_queue()			

		self.gcode_textbox.on_submit(submit)

	def run_queue(self):
		start_time = time.time()

		# if no moves to make, return
		if len(self.mot.queue) == 0:
			return

		final_joint_angles = self.mot.queue[-1]

		# render with artists
		self.clear()
		artists = []
		temp_kin = Kinematics()
		while len(self.mot.queue) != 0:
			# update joint angles of kinematic model
			temp_kin.update_joint_angles(self.mot.queue.pop(0))
			temp_kin.calc_t_matrices()

			# get arm lines
			self.draw_arm(temp_kin)

			# add lines for this frame to artists array
			artists.append(self.ax.lines)
			self.clear()
		temp_kin = None

		print("Finished rendering in", time.time()-start_time, "seconds")

		self.ani = ArtistAnimation(self.fig, artists, interval=30, repeat=False, blit=True)

		# wait for animation to finish, then render final frame
		plt.pause(((30+5)/1000)*len(artists))
		self.kin.update_joint_angles(final_joint_angles)
		self.kin.calc_t_matrices()
		self.clear()
		self.draw_arm(self.kin)
		self.render()

	def draw_arm(self, kin):
		'''
		Draw the arm

		Parameters
		  kin: kinematic model to draw
		'''

		# draw arm
		self.draw_line(np.array([0,0,0]), 	 kin.t_matrix_0_1.pos)
		self.draw_line(kin.t_matrix_0_1.pos, kin.t_matrix_0_2.pos)
		self.draw_line(kin.t_matrix_0_2.pos, kin.t_matrix_0_3.pos)
		self.draw_line(kin.t_matrix_0_3.pos, kin.t_matrix_0_4.pos)
		self.draw_line(kin.t_matrix_0_4.pos, kin.t_matrix_0_5.pos)
		self.draw_line(kin.t_matrix_0_5.pos, kin.t_matrix_0_6.pos)

		# draw rotation of end effector
		self.draw_axies(kin.t_matrix_0_1.pos, kin.t_matrix_0_1.rot)
		self.draw_axies(kin.t_matrix_0_2.pos, kin.t_matrix_0_2.rot)
		self.draw_axies(kin.t_matrix_0_3.pos, kin.t_matrix_0_3.rot)
		self.draw_axies(kin.t_matrix_0_4.pos, kin.t_matrix_0_4.rot)
		self.draw_axies(kin.t_matrix_0_5.pos, kin.t_matrix_0_5.rot)
		self.draw_axies(kin.t_matrix_0_6.pos, kin.t_matrix_0_6.rot)

	def clear(self):
		'''
		Remove all lines from plot
		'''

		self.ax.lines = []

	def render(self):
		'''
		Render the lines drawn
		'''
		self.set_axes_equal()
		plt.draw()

	def show(self):
		'''
		Open window
		'''
		plt.show()