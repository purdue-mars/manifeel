import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading

bridge = CvBridge()

# Global variables for images
side_image = None
front_image = None
wrist_image = None
wrist2_image = None
tactile_left_image = None
tactile_right_image = None
tactile_shear_left_image = None
tactile_shear_right_image = None

# Mutex to prevent race conditions
lock = threading.Lock()

# Function to convert the image's color order (e.g., BGR to RGB or vice versa)
def correct_color_order(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Adjust as needed, e.g., cv2.COLOR_RGB2BGR

def image_callback_side(msg):    
    global side_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        side_image = correct_color_order(raw_image)

def image_callback_front(msg):    
    global front_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        front_image = correct_color_order(raw_image)

def image_callback_wrist(msg):    
    global wrist_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        wrist_image = correct_color_order(raw_image)

def image_callback_wrist2(msg):    
    global wrist2_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        wrist2_image = correct_color_order(raw_image)

def image_callback_tactile_left(msg):    
    global tactile_left_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        tactile_left_image = correct_color_order(raw_image)

def image_callback_tactile_right(msg):    
    global tactile_right_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        tactile_right_image = correct_color_order(raw_image)

def image_callback_tactile_shear_left(msg):    
    global tactile_shear_left_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # tactile_shear_left_image = correct_color_order(raw_image)
        tactile_shear_left_image = raw_image

def image_callback_tactile_shear_right(msg):    
    global tactile_shear_right_image
    with lock:
        raw_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # tactile_shear_right_image = correct_color_order(raw_image)
        tactile_shear_right_image = raw_image

def display_images():
    while not rospy.is_shutdown():
        with lock:
            if side_image is not None:
                cv2.imshow('Side Image', side_image)
            if front_image is not None:
                cv2.imshow('Front Image', front_image)
            if wrist_image is not None:
                cv2.imshow('Wrist Image', wrist_image)
            if wrist2_image is not None:
                cv2.imshow('Wrist2 Image', wrist2_image)
            if tactile_left_image is not None:
                cv2.imshow('Tactile Left Image', tactile_left_image)
            if tactile_right_image is not None:
                cv2.imshow('Tactile Right Image', tactile_right_image)
            if tactile_shear_left_image is not None:
                cv2.imshow('Tactile Shear Left Image', tactile_shear_left_image)
            if tactile_shear_right_image is not None:
                cv2.imshow('Tactile Shear Right Image', tactile_shear_right_image)

        cv2.waitKey(1)

# ROS Initialization
rospy.init_node('image_subscriber', anonymous=True)

# Subscribe to image topics
rospy.Subscriber('/camera/side_image', Image, image_callback_side)
rospy.Subscriber('/camera/front_image', Image, image_callback_front)
rospy.Subscriber('/camera/wrist_image', Image, image_callback_wrist)
# rospy.Subscriber('/camera/wrist2_image', Image, image_callback_wrist2)
# rospy.Subscriber('/camera/tactile_shear_left_image', Image, image_callback_tactile_shear_left)
rospy.Subscriber('/camera/tactile_left_image', Image, image_callback_tactile_left)
rospy.Subscriber('/camera/tactile_shear_right_image', Image, image_callback_tactile_shear_right)
rospy.Subscriber('/camera/tactile_right_image', Image, image_callback_tactile_right)

# Start image display in the main thread
display_images()

# import rospy
# import cv2
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# '''
# def image_callback(msg):
#     bridge = CvBridge()
    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
#     cv2.imshow('Received Side Image', cv_image)
#     cv2.waitKey(1)  

# rospy.init_node('image_subscriber', anonymous=True)
# rospy.Subscriber('/camera/side_image', Image, image_callback)

# rospy.spin()
# '''

# bridge = CvBridge()

# def image_callback_side(msg):    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#     cv2.imshow('Side Image', cv_image)
#     cv2.waitKey(1)

# def image_callback_front(msg):    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#     cv2.imshow('Front Image', cv_image)
#     cv2.waitKey(1)

# def image_callback_wrist(msg):    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#     cv2.imshow('Wrist Image', cv_image)
#     cv2.waitKey(1)

# def image_callback_wrist2(msg):    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#     cv2.imshow('Wrist2 Image', cv_image)
#     cv2.waitKey(1)

# def image_callback_tactile_left(msg):    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#     cv2.imshow('Tactile Left Image', cv_image)
#     cv2.waitKey(1)

# def image_callback_tactile_right(msg):    
#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#     cv2.imshow('Tactile Right Image', cv_image)
#     cv2.waitKey(1)

# rospy.init_node('image_subscriber', anonymous=True)

# #rospy.Subscriber('/camera/side_image', Image, image_callback_side)
# rospy.Subscriber('/camera/front_image', Image, image_callback_front)
# rospy.Subscriber('/camera/wrist_image', Image, image_callback_wrist)
# #rospy.Subscriber('/camera/wrist2_image', Image, image_callback_wrist2)
# #rospy.Subscriber('/camera/tactile_left_image', Image, image_callback_tactile_left)
# #rospy.Subscriber('/camera/tactile_right_image', Image, image_callback_tactile_right)

# rospy.spin()

'''
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading

bridge = CvBridge()

# Global variables for images
side_image = None
front_image = None
wrist_image = None
wrist2_image = None
tactile_left_image = None
tactile_right_image = None

# Mutex to prevent race conditions
lock = threading.Lock()

def image_callback_side(msg):    
    global side_image
    with lock:
        side_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def image_callback_front(msg):    
    global front_image
    with lock:
        front_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def image_callback_wrist(msg):    
    global wrist_image
    with lock:
        wrist_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def image_callback_wrist2(msg):    
    global wrist2_image
    with lock:
        wrist2_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def image_callback_tactile_left(msg):    
    global tactile_left_image
    with lock:
        tactile_left_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def image_callback_tactile_right(msg):    
    global tactile_right_image
    with lock:
        tactile_right_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def display_images():
    while not rospy.is_shutdown():
        with lock:
            if side_image is not None:
                cv2.imshow('Side Image', side_image)
            if front_image is not None:
                cv2.imshow('Front Image', front_image)
            if wrist_image is not None:
                cv2.imshow('Wrist Image', wrist_image)
            if wrist2_image is not None:
                cv2.imshow('Wrist2 Image', wrist2_image)
            if tactile_left_image is not None:
                cv2.imshow('Tactile Left Image', tactile_left_image)
            if tactile_right_image is not None:
                cv2.imshow('Tactile Right Image', tactile_right_image)

        cv2.waitKey(1)

# ROS Initialization
rospy.init_node('image_subscriber', anonymous=True)

# Subscribe to image topics
rospy.Subscriber('/camera/side_image', Image, image_callback_side)
rospy.Subscriber('/camera/front_image', Image, image_callback_front)
rospy.Subscriber('/camera/wrist_image', Image, image_callback_wrist)
# rospy.Subscriber('/camera/wrist2_image', Image, image_callback_wrist2)
# rospy.Subscriber('/camera/tactile_left_image', Image, image_callback_tactile_left)
# rospy.Subscriber('/camera/tactile_right_image', Image, image_callback_tactile_right)

# Start image display in the main thread
display_images()
'''