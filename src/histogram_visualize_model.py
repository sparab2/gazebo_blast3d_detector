import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gazebo_blast3d.msg import EventArray
from collections import deque
from rospy.timer import Rate
import threading

class DiffusionModel:
    def __init__(self, width, height, D, dt):
        self.width = width
        self.height = height
        self.D = D  # Diffusion coefficient
        self.dt = dt  # Time step
        self.density = np.zeros((height, width))

    def update_density(self):
        # Calculate gradients twice to simulate the second derivative (laplacian)
        dy, dx = np.gradient(self.density)
        d2y = np.gradient(dy, axis=0)  # Second derivative with respect to y
        d2x = np.gradient(dx, axis=1)  # Second derivative with respect to x

        # Update density using the diffusion equation
        self.density += self.D * self.dt * (d2x + d2y)

    def add_event(self, x, y):
        # Simple model to add an event at (x, y)
        if 0 <= x < self.width and 0 <= y < self.height:
            self.density[y, x] += 1

    def decay_events(self):
        # Simple decay of events over time
        self.density *= 0.99

class EventStream:
    def __init__(self, max_length=1000, height=480, blast_threshold=20):
        self.y = deque(maxlen=max_length)  # Using deque for automatic handling of max length
        self.height = height
        self.event_histories = []
        self.blast_threshold = blast_threshold 

        # Initialize ROS Node and Subscriber
        rospy.init_node('event_stream')
        rospy.Subscriber('/event_topic', EventArray, self.callback)

    def callback(self, data):
        for event in data.events:
            self.add_event(event.y)  # Assuming 'y' is a property of events in EventArray

    def add_event(self, y):
        self.y.append(y)
        #print(f"Added y-coordinate: {y}, Total events: {len(self.y)}")

    def detect_blasts(self, histogram):
        if len(self.event_histories) > 10:  # Ensure there's enough data to analyze
            recent_histograms = np.array(self.event_histories[-10:])  # Last 10 histograms
            avg_events = np.mean(recent_histograms, axis=0)
            std_dev = np.std(recent_histograms, axis=0)
            threshold = avg_events + 2 * std_dev  # Setting threshold as mean + 2*std

            peaks = np.sum(histogram > threshold)
            if peaks > 4:
                print(f"[ALERT] Blast detected with {peaks} high-density areas exceeding thresholds!")


    def update_histogram(self):
        y_array = np.array(self.y)
        histogram, _ = np.histogram(y_array, bins=30, range=(0, self.height))
        self.event_histories.append(histogram)
        self.detect_blasts(histogram)

class ROSHistogramVisualizer:
    def __init__(self, event_stream):
        self.event_stream = event_stream
        self.fig, self.ax = plt.subplots()

    def update_plot(self, frame):
        self.ax.clear()
        y_array = np.array(self.event_stream.y)
        if len(y_array) == 0:
            print("No data to plot.")
            return
        
        histogram, bin_edges = np.histogram(y_array, bins=30, range=(0, 480))
        self.ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align='edge', color='blue')
        self.ax.set_title("Event Distribution Histogram")
        self.ax.set_xlabel('Vertical Position')
        self.ax.set_ylabel('Number of Events')
        plt.draw()

    def run(self):
        # Start a separate thread for ROS operations to prevent blocking by plt.show()
        def ros_thread():
            rate = Rate(10)  # 10 Hz
            while not rospy.is_shutdown():
                self.event_stream.update_histogram()
                rate.sleep()

        t = threading.Thread(target=ros_thread)
        t.start()

        ani = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        plt.show()
        t.join()  # Ensure ROS thread finishes cleanly after closing the plot

if __name__ == '__main__':
    event_stream = EventStream(max_length=1000, height=480)
    visualizer = ROSHistogramVisualizer(event_stream)
    visualizer.run()
