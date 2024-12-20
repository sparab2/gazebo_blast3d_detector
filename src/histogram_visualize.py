import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gazebo_blast3d.msg import EventArray
from scipy.signal import find_peaks

class EventStream:
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.x = []
        self.y = []
        self.t = []
        self.p = []

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

class ROSHistogramVisualizer:
    def __init__(self):
        rospy.init_node('event_histogram_visualizer', anonymous=True)
        self.event_stream = EventStream()
        self.subscriber = rospy.Subscriber('/event_topic', EventArray, self.event_stream.callback)
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Y Bins')
        self.ax.set_ylabel('Event Count')
        self.ax.set_title('Histogram of Y Coordinate Events')

    def update_plot(self, frame):
        if not self.event_stream.y:  # If no data, skip
            return

        # Calculate histogram
        y_array = np.array(self.event_stream.y)
        histogram, bin_edges = np.histogram(y_array, bins=24, range=(0, 480))

        # Dynamic threshold based on the median or mean
        dynamic_threshold = np.median(histogram) + 1.5 * np.std(histogram)  # Adjust factor based on your data characteristics

        # Find peaks in the histogram with dynamic threshold
        peaks, properties = find_peaks(histogram, height=dynamic_threshold, prominence=100, width=3)  # Adjust prominence and width as needed

        # Update the plot
        self.ax.cla()  # Clear current axes
        self.ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align='edge')
        self.ax.plot(bin_edges[:-1][peaks], histogram[peaks], "x", color='red')  # Mark the peaks with 'x'

        # Print peak properties
        print("Peak Properties:")
        print("Heights:", properties["peak_heights"])
        print("Prominences:", properties["prominences"])
        print("Widths:", properties["widths"])
        print("Left Bases (indices):", properties["left_bases"])
        print("Right Bases (indices):", properties["right_bases"])
        print("Left Interpolated Positions:", properties["left_ips"])
        print("Right Interpolated Positions:", properties["right_ips"])

        self.ax.set_xlabel('Y Bins')
        self.ax.set_ylabel('Event Count')
        self.ax.set_ylim(0, histogram.max() + 1)  # Adjust y-limits for visibility

        plt.draw()

    def run(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

if __name__ == '__main__':
    visualizer = ROSHistogramVisualizer()
    visualizer.run()
