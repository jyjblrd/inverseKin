import numpy as np
from renderer import Renderer
from kinematics import Kinematics, TMatrix
from motionPlanner import MotionPlanner
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

kin = Kinematics()
mot = MotionPlanner()

rend = Renderer(kin, mot)
rend.draw_joint_sliders()
rend.draw_pos_sliders()
rend.draw_text_box()
rend.draw_arm(kin)
rend.render()

mot.file("test.gcode")
rend.run_queue()

rend.show()
plt.show()