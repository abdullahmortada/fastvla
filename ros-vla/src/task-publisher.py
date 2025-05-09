import rospy
from std_msgs.msg import String 

class TaskPublisher:
    def __init__(self):
        rospy.init_node('task-publisher', anonymous=True)

        self.rate = rospy.Rate(10)
        self.taskPub = rospy.Publisher('/task_string', String)

    def run(self):
        while not rospy.is_shutdown():
            taskString = input("What do you want the robot to do next?\n")
            self.taskPub.publish(taskString)
            self.rate.sleep()
            
if __name__ == '__main__':
    try:
        publisher = TaskPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
