import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
    
    #sub하는 시점에 실행되는 함수 -> detection이는 토픽이 수신되면! 
    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        
        if len(detection_msg.detections) == 0:
            return
        
        lane2_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane2', color=255) #0~255 밝기정도, 255:흰색

        (h, w) = (lane2_edge_image.shape[0], lane2_edge_image.shape[1]) #(480, 640)
        dst_mat = [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], [round(w * 0.7), h], [round(w * 0.3), h]]
        src_mat = [[238, 316],[402, 313], [501, 476], [155, 476]]
        
        #bird : 위에서 아래를 바라보는 각도
        lane2_bird_image = CPFL.bird_convert(lane2_edge_image, srcmat=src_mat, dstmat=dst_mat)
        #roi: 차랑 가까운 도로부분만 보도록 픽셀을 자름
        roi_image = CPFL.roi_rectangle_below(lane2_bird_image, cutting_idx=300) 

        if self.show_image:
            cv2.imshow('lane2_edge_image', lane2_edge_image)
            cv2.imshow('lane2_bird_img', lane2_bird_image)
            cv2.imshow('roi_img', roi_image)

            cv2.waitKey(1)
            
        #차선 기울기 반환하는 함수
        grad = CPFL.dominant_gradient(roi_image, theta_limit=70) #원래 보통 차선은 직선이므로, 가로선이 들어오면 그건 노이즈로 받음. 70 이상(각도)이 들어오면 취급하지 않겠다. 최대한 꺾어도 20정도임
        
        #target_point: 두 차선에 가로선을 그었을 때 두 교점의 중점
        target_point_y = 90 #가로선이 높을 수도, 낮을 수도 있음 -> 값으로 높이 지정. 작아질 수록 위로 가고, 커질 수록 밑으로 감
        target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, 
                                            detection_thickness=10, road_gradient=grad, lane_width=300)
        
        # detection_thickness: 차선이 노이즈로 끊겨져있을 때 타겟 검출선의 두께를 정하는 파라미터. 두께가 늘어나면 -> 교점이 늘어남
        # lane_width: 카메라에 차선이 하나만 인식될 때, 반대편에 그려줌. 차선 폭이 300픽셀로 지정했으니, 교점의 150픽셀 옆이 중점이라 생각하는 것.

        lane = LaneInfo()
        lane.slope = grad
        lane.target_x = round(target_point_x)
        lane.target_y = round(target_point_y)

        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()
