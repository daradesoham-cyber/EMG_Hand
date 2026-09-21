import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class GesturePublisher(Node):

    def __init__(self):
        super().__init__('gesture_publisher')

        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/hand_servo_input',
            10
        )

        self.gesture = 0

        self.timer = self.create_timer(
            4.0,
            self.publish_gesture
        )

        self.get_logger().info(
            'EMG hand demo started.'
        )
        self.get_logger().info(
            'REST -> FIST -> OPEN'
        )

    def publish_gesture(self):

        if self.gesture == 0:

            # REST / OPEN
            pose = [0.0] * 16
            name = 'REST / OPEN'

        elif self.gesture == 1:

            # Gentle fist demonstration
            pose = [25.0] * 16
            name = 'FIST'

        else:

            # Intermediate open pose
            pose = [5.0] * 16
            name = 'OPEN'

        msg = Float32MultiArray()
        msg.data = pose

        self.publisher.publish(msg)

        self.get_logger().info(
            f'Publishing gesture: {name}'
        )

        self.gesture = (self.gesture + 1) % 3


def main(args=None):

    rclpy.init(args=args)

    node = GesturePublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
