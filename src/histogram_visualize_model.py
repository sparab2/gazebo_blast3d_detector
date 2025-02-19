import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gazebo_blast3d.msg import EventArray
from collections import deque
from rospy.timer import Rate
import time

import threading

class EventStream:
    def __init__(self, max_length=1000, height=480, width=640, blast_threshold=8, verification_threshold=2, min_data_threshold=40):
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
        self.last_receive_time = None
        self.latency_measurements = []  # List to store latency measurements
        
        rospy.init_node('event_stream')
        
        rospy.Subscriber('/event_topic', EventArray, self.callback)
        rospy.on_shutdown(self.calculate_average_latency)  # Register shutdown hook

    def callback(self, data):
        receive_time = time.time()  # Record time when data is received
        #self.last_receive_time = receive_time
        for event in data.events:
            self.add_event(event.x, event.y)
        self.update_histogram(receive_time)
        
    def add_event(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def update_histogram(self, receive_time):
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
                    current_time = time.time()  # Timestamp when blast is confirmed
                    latency = current_time - receive_time
                    self.latency_measurements.append(latency)  # Store latency
                    rospy.loginfo(f"[ALERT] Blast detected! Latency: {latency:.3f} seconds")
                    self.consecutive_detections = 0
                return True
            else:
                self.consecutive_detections = 0
        return False

    def apply_diffusion(self):
        if len(self.event_histories_y) > 1:
            self.event_histories_y[-1] = (self.event_histories_y[-2] * 0.8 + self.event_histories_y[-1] * 0.2).astype(int)
            self.event_histories_x[-1] = (self.event_histories_x[-2] * 0.8 + self.event_histories_x[-1] * 0.2).astype(int)

    def confirm_blast_y(self, histogram_y):
        recent_history_y = np.array(self.event_histories_y[-5:])
        avg_events_y = np.mean(recent_history_y, axis=0)
        std_dev_y = np.std(recent_history_y, axis=0)
        threshold_y = avg_events_y + 1.8 * std_dev_y
        #print(f"Y-Avg: {avg_events_y}, Y-Std: {std_dev_y}, Y-Threshold: {threshold_y}")  # Debug info
        result = np.any(histogram_y > threshold_y)
        #if result:
         #   print("Blast detected on Y-axis.")
        return result

    def confirm_blast_x(self, histogram_x):
        recent_history_x = np.array(self.event_histories_x[-5:])
        avg_events_x = np.mean(recent_history_x, axis=0)
        std_dev_x = np.std(recent_history_x, axis=0)
        threshold_x = avg_events_x + 1.4 * std_dev_x
        #print(f"X-Avg: {avg_events_x}, X-Std: {std_dev_x}, X-Threshold: {threshold_x}")  # Debug info
        result = np.any(histogram_x > threshold_x)
        #if result:
         #   print("Blast detected on X-axis.")
        return result

    def calculate_average_latency(self):
        if self.latency_measurements:
            average_latency = sum(self.latency_measurements) / len(self.latency_measurements)
            rospy.loginfo(f"Average Latency: {average_latency:.3f} seconds")

class ROSHistogramVisualizer:
    def __init__(self, event_stream):
        self.event_stream = event_stream
        self.fig, self.axs = plt.subplots(2, 1)  # Create two subplots vertically aligned
        self.last_below_threshold_y = True
        self.last_below_threshold_x = True

    '''
    def update_plot(self, frame):
        y_array = np.array(self.event_stream.y)
        x_array = np.array(self.event_stream.x)
        histogram_y, bin_edges_y = np.histogram(y_array, bins=50, range=(0, 480))
        histogram_x, bin_edges_x = np.histogram(x_array, bins=50, range=(0, 640))

        self.axs[0].cla()
        self.axs[1].cla()

        # Highlight the bars where blasts are detected
        bar_colors_y = ['blue' if value <= self.event_stream.threshold_y[-1] else 'green' for value in histogram_y]
        bar_colors_x = ['red' if value <= self.event_stream.threshold_x[-1] else 'green' for value in histogram_x]

        self.axs[0].bar(bin_edges_y[:-1], histogram_y, width=np.diff(bin_edges_y), align='edge', color=bar_colors_y)
        self.axs[1].bar(bin_edges_x[:-1], histogram_x, width=np.diff(bin_edges_x), align='edge', color=bar_colors_x)

        self.axs[0].set_title("Y-axis Event Distribution")
        self.axs[1].set_title("X-axis Event Distribution")
        plt.tight_layout()

    '''
    def update_plot(self, frame):
        if not self.event_stream.y:
            return

        y_array = np.array(self.event_stream.y)
        x_array = np.array(self.event_stream.x)

        histogram_y, bin_edges_y = np.histogram(y_array, bins=50, range=(0, self.event_stream.height))
        histogram_x, bin_edges_x = np.histogram(x_array, bins=50, range=(0, self.event_stream.width))

        self.axs[0].cla()  # Clear the Y-axis subplot
        self.axs[1].cla()  # Clear the X-axis subplot

        rects_y = self.axs[0].bar(bin_edges_y[:-1], histogram_y, width=np.diff(bin_edges_y), align='edge', color='blue')
        rects_x = self.axs[1].bar(bin_edges_x[:-1], histogram_x, width=np.diff(bin_edges_x), align='edge', color='red')

        self.annotate_blasts(rects_y, histogram_y, self.axs[0], self.event_stream.confirm_blast_y, 'y')
        self.annotate_blasts(rects_x, histogram_x, self.axs[1], self.event_stream.confirm_blast_x, 'x')

        self.axs[0].set_title("Y-axis Event Distribution")
        self.axs[0].set_xlabel('Vertical Position')
        self.axs[0].set_ylabel('Number of Events')
        self.axs[1].set_title("X-axis Event Distribution")
        self.axs[1].set_xlabel('Horizontal Position')
        self.axs[1].set_ylabel('Number of Events')

        plt.tight_layout()  # Adjust subplots to fit into figure cleanly

    

    def annotate_blasts(self, rects, histogram, ax, confirm_func, axis):
        last_below_threshold = True
        for i, (rect, value) in enumerate(zip(rects, histogram)):
            if confirm_func([value]):  # Assumes confirm_func can handle single-value lists
                if last_below_threshold:
                    ax.annotate('Start', (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                textcoords="offset points", xytext=(0,10), ha='center', color='green')
                    last_below_threshold = False
            else:
                last_below_threshold = True


    def run(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        plt.show()

if __name__ == '__main__':
    event_stream = EventStream(max_length=1000, height=480)
    visualizer = ROSHistogramVisualizer(event_stream)
    threading.Thread(target=visualizer.run).start()
    rospy.spin()
