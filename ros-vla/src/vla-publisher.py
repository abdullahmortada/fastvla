import numpy as np 
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation

import rospy
import tf2_ros
import actionlib
import franka_gripper.msg
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String  # For gripper control (if needed)
from sensor_msgs.msg import Image

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_model
)


class VLAActionPublisher:
    def __init__(self, max_steps=1000):
        self.image = np.array([None])
        self.bridge = CvBridge()

        self.max_steps = max_steps
        self.task_string = " "
        self.steps = self.max_steps
        self.steps = 0

        self.move_speed = 0.2

        # === OpenVLA initialization ===
        self.model_family = "openvla"                   # Model family
        self.pretrained_checkpoint = "/home/lab/ModelWeights/openvla-7b-finetuned-libero-spatial"                 # Pretrained checkpoint path
        self.load_in_8bit = False                       # (For OpenVLA only) Load with 8-bit quantization
        self.load_in_4bit = True                       # (For OpenVLA only) Load with 4-bit quantization

        self.center_crop = True                         # Center crop? (if trained w/ random crop image aug)

        self.unnorm_key = "libero_spatial"
        self.model = get_model(self)
        self.processor = None
        if self.model_family == "openvla":
            self.processor = get_processor(self)
            print("get_processor()")

        self.resize_size = 224


        # === ROS Initialization ===
        rospy.init_node('vla-action-publisher', anonymous=True)

        self.rate = rospy.Rate(10)  # 10 Hz

        print("Ros node initialized")

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.cartesian_pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)

        rospy.Subscriber('/task_string', String, self.task_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)

        self.gripperClient = actionlib.SimpleActionClient('/franka_gripper/move', franka_gripper.msg.MoveAction)
        self.gripperClient.wait_for_server()

        print("Initialized object instance")


    # callback to change current task, will subscribe to text node for tasks 
    def task_callback(self, data):
        self.steps = 0 
        self.task_string = data.data
    
    def camera_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = np.array(frame, dtype=np.uint8)
        self.image = cv2.resize(frame, dsize=(self.resize_size, self.resize_size), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("./image_last.jpg", self.image)
        

    def run(self):
        while not rospy.is_shutdown():
            if self.steps >= self.max_steps:
                print("Past max steps for task")
                continue

            if self.image.any() == None: 
                continue

            observation = self.get_observation()

            action = get_action(self, self.model, observation, self.task_string, processor=self.processor)
            print("Action: " + str(action))

            # Convert the action to movement commands
            pose, gripper_msg = self.action_to_commands(action)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "panda_link0"
            pose_msg.pose = pose

            self.cartesian_pub.publish(pose_msg)
            gripperGoal = franka_gripper.msg.MoveGoal(width=gripper_msg, speed=1.0)
            self.gripperClient.send_goal(gripperGoal)

            self.steps += 1

            # Sleep to maintain the loop rate
            self.rate.sleep()
            

    def get_observation(self):
        pose = self.get_current_pose()
        rot = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        rot = rot.as_euler('xyz')
        obs = {
            "full_image": self.image,
            "state": [pose.position.x, pose.position.y, pose.position.z, rot[0], rot[1], rot[2], 0]
        }
        return obs  # Replace with actual observation

    def action_to_commands(self, action):
        rotdelta = Rotation.from_euler('xyz', [a * self.move_speed for a in action[3:6]])
        quatdelta = rotdelta.as_quat()

        # quatdelta = axisangle2quat([a * self.move_speed for a in [action[3], action[4], action[5]] ])
        pose = self.get_current_pose()
        rot = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        rotgoal = rot * quatdelta
        rotgoal = rotgoal.as_quat()

        pose.position.x += (action[0] * self.move_speed)  
        pose.position.y += (action[1]  * self.move_speed) 
        pose.position.z += (action[2]  * self.move_speed) 
        pose.orientation.x = rotgoal[0]  
        pose.orientation.y = rotgoal[1]  
        pose.orientation.z = rotgoal[2]  
        pose.orientation.w = rotgoal[3]  
        print("pose to go: \n", pose, end="\n\n")

        gripper_msg = action[6]  # Gripper opening value


        return pose, gripper_msg

    def get_current_pose(self):
        try:
            trans = self.tfBuffer.lookup_transform('panda_link0', 'panda_hand', rospy.Time(0), rospy.Duration(1.0))
            print(trans)
            pose = Pose()
            pose.position.x = trans.transform.translation.x
            pose.position.y = trans.transform.translation.y
            pose.position.z = trans.transform.translation.z
            pose.orientation = trans.transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
        
        return pose

if __name__ == '__main__':
    try:
        print("Initializing")
        publisher = VLAActionPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
