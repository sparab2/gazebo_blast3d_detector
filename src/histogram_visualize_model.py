import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gazebo_blast3d.msg import EventArray
from collections import deque
from rospy.timer import Rate

import threading

class EventStream:
    def __init__(self, max_length=1000, height=480, width=640, blast_threshold=10, verification_threshold=3, min_data_threshold=50):
        self.x = deque(maxlen=max_length)
        self.y = deque(maxlen=max_length)
        self.height = height
        self.width = width
        self.event_histories_y = []
        self.event_histories_x = []
        self.blast_threshold = blast_threshold
        self.verification_threshold = verification_threshold
        self.min_data_threshold = min_data_threshold
        self.consecutive_detections = 0

        rospy.init_node('event_stream')
        rospy.Subscriber('/event_topic', EventArray, self.callback)

    def callback(self, data):
        for event in data.events:
            self.add_event(event.x, event.y)
        self.update_histogram()

    def add_event(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def update_histogram(self):
        if len(self.y) > self.min_data_threshold:
            y_array = np.array(self.y)
            x_array = np.array(self.x)
            histogram_y, _ = np.histogram(y_array, bins=20, range=(min(self.y), max(self.y)))
            histogram_x, _ = np.histogram(x_array, bins=20, range=(min(self.x), max(self.x)))
            self.event_histories_y.append(histogram_y)
            self.event_histories_x.append(histogram_x)
            self.apply_diffusion()

            if self.confirm_blast_y(self.event_histories_y[-1]) and self.confirm_blast_x(self.event_histories_x[-1]):
                self.consecutive_detections += 1
                if self.consecutive_detections >= self.verification_threshold:
                    print("[ALERT] Confirmed blast detection with significant activity on both axes!")
                    self.consecutive_detections = 0
                else:
                    self.consecutive_detections = 0

    def apply_diffusion(self):
        if len(self.event_histories_y) > 1:
            self.event_histories_y[-1] = (self.event_histories_y[-2] * 0.8 + self.event_histories_y[-1] * 0.2).astype(int)
            self.event_histories_x[-1] = (self.event_histories_x[-2] * 0.8 + self.event_histories_x[-1] * 0.2).astype(int)

    def confirm_blast_y(self, histogram_y):
        recent_history_y = np.array(self.event_histories_y[-5:])
        avg_events_y = np.mean(recent_history_y, axis=0)
        threshold_y = avg_events_y + 1.96 * np.std(recent_history_y, axis=0)
        return np.any(histogram_y > threshold_y)

    def confirm_blast_x(self, histogram_x):
        recent_history_x = np.array(self.event_histories_x[-5:])
        avg_events_x = np.mean(recent_history_x, axis=0)
        threshold_x = avg_events_x + 1.5 * np.std(recent_history_x, axis=0)
        return np.any(histogram_x > threshold_x)

class ROSHistogramVisualizer:
    def __init__(self, event_stream):
        self.event_stream = event_stream
        self.fig, self.axs = plt.subplots(2, 1)  # Create two subplots vertically aligned

    def update_plot(self, frame):
        if not self.event_stream.y:
            return

        y_array = np.array(self.event_stream.y)
        x_array = np.array(self.event_stream.x)
        
        histogram_y, bin_edges_y = np.histogram(y_array, bins=50, range=(0, 480))
        histogram_x, bin_edges_x = np.histogram(x_array, bins=50, range=(0, 640))
        
        self.axs[0].cla()  # Clear the Y-axis subplot
        self.axs[1].cla()  # Clear the X-axis subplot
        
        self.axs[0].bar(bin_edges_y[:-1], histogram_y, width=np.diff(bin_edges_y), align='edge', color='blue')
        self.axs[0].set_title("Y-axis Event Distribution")
        self.axs[0].set_xlabel('Vertical Position')
        self.axs[0].set_ylabel('Number of Events')

        self.axs[1].bar(bin_edges_x[:-1], histogram_x, width=np.diff(bin_edges_x), align='edge', color='red')
        self.axs[1].set_title("X-axis Event Distribution")
        self.axs[1].set_xlabel('Horizontal Position')
        self.axs[1].set_ylabel('Number of Events')

        plt.tight_layout()  # Adjust subplots to fit into figure cleanly

    def run(self):
        def ros_thread():
            rate = Rate(10)  # 10 Hz
            while not rospy.is_shutdown():
                self.event_stream.update_histogram()
                rate.sleep()

        threading.Thread(target=ros_thread).start()
        ani = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        plt.show()

if __name__ == '__main__':
    event_stream = EventStream(max_length=1000, height=480)
    visualizer = ROSHistogramVisualizer(event_stream)
    visualizer.run()
