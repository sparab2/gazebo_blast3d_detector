import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from gazebo_blast3d.msg import EventArray
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class EventStream:
    def __init__(self, max_length=1000):
        self.max_length = max_length  # Maximum number of points to display
        self.x = []
        self.y = []
        self.t = []
        self.p = []  # Store polarity

    def callback(self, data):
        print(f"Received {len(data.events)} events")
        current_time = rospy.Time.now().to_sec()
        new_x = []
        new_y = []
        new_t = []
        new_p = []

        for event in data.events:
            new_x.append(event.x)
            new_y.append(event.y)
            new_t.append(current_time)
            new_p.append(event.polarity)

        # Add new data in a way that keeps all lists synchronized
        if len(self.t) + len(new_t) > self.max_length:
            overflow = len(self.t) + len(new_t) - self.max_length
            self.x = self.x[overflow:] + new_x
            self.y = self.y[overflow:] + new_y
            self.t = self.t[overflow:] + new_t
            self.p = self.p[overflow:] + new_p
        else:
            self.x.extend(new_x)
            self.y.extend(new_y)
            self.t.extend(new_t)
            self.p.extend(new_p)
    def calculate_histogram(self, bin_count=24):  # Adjust bin_count as necessary
        if self.y:  # Check if there are y-coordinates to process
            y_array = np.array(self.y)
            histogram, bin_edges = np.histogram(y_array, bins=bin_count, range=(0, 480))
            return histogram, bin_edges
        return None, None

class ROSVisualizer:
    def __init__(self):
        rospy.init_node('event_visualizer', anonymous=True)
        self.event_stream = EventStream()
        self.subscriber = rospy.Subscriber('/event_topic', EventArray, self.event_stream.callback)
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.sc = None

        # Setting the labels for each axis
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Time (ms)')
        self.ax.set_zlabel('Y Axis')

        # Set the view of the 3D plot
        self.ax.view_init(elev=10, azim=60)  # Adjust these angles to get the desired perspective

    def update_plot(self, frame):
        if not self.event_stream.x:  # If no data, skip
            return

        xs = np.array(self.event_stream.x)
        ys = np.array(self.event_stream.y)
        ts = np.array(self.event_stream.t)
        ps = np.array(self.event_stream.p)

        # Ensure all arrays are the same length before plotting
        if not (len(xs) == len(ys) == len(ts) == len(ps)):
            print("Array length mismatch")
            return

        colors = np.where(ps, 'blue', 'red')
        self.ax.cla()  # Clear the current axes

        self.sc = self.ax.scatter(-xs, ts, -ys, c=colors, depthshade=True)
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Time (ms)')
        self.ax.set_zlabel('Y Axis')

        # Set limits
        self.ax.set_xlim([min(-xs), max(-xs)] if len(xs) > 0 else [-1, 1])
        self.ax.set_ylim([min(ts), max(ts)] if len(ts) > 0 else [-1, 1])
        self.ax.set_zlim([min(-ys), max(-ys)] if len(ys) > 0 else [-1, 1])

        plt.draw()

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

if __name__ == '__main__':
    visualizer = ROSVisualizer()
    visualizer.run()
