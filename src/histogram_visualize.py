import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gazebo_blast3d.msg import EventArray
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from collections import deque

class EventStream:
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.x = deque(maxlen=max_length)
        self.y = deque(maxlen=max_length)
        self.t = deque(maxlen=max_length)
        self.p = deque(maxlen=max_length)
        self.event_rate_history = deque(maxlen=50)
        self.blast_counter = 0
        self.y_expansion_threshold = 100
        self.x_expansion_threshold = 80

    def update_event_rate_threshold(self):
        if len(self.event_rate_history) < 10:
            return 20  # Lower default threshold
        return np.mean(self.event_rate_history) + 0.5 * np.std(self.event_rate_history)

    def get_temporal_changes(self, event_count):
        self.event_rate_history.append(event_count)

        if len(self.event_rate_history) > 1:
            rate_change = event_count - self.event_rate_history[-2]
            dynamic_threshold = self.update_event_rate_threshold()

            #print(f"[DEBUG] Event rate change: {rate_change}, dynamic threshold: {dynamic_threshold}")

            if rate_change > dynamic_threshold:
                self.blast_counter += 1
                #print(f"[DEBUG] Potential blast #{self.blast_counter} detected due to event rate spike.")
                return True
        return False

    def callback(self, data):
        current_time = rospy.Time.now().to_sec()
        new_x, new_y, new_t, new_p = [], [], [], []
        for event in data.events:
            new_x.append(event.x)
            new_y.append(event.y)
            new_t.append(current_time)
            new_p.append(event.polarity)
        self.extend_data(new_x, new_y, new_t, new_p)

        # Calculate the number of events received in this batch
        event_count = len(data.events)
        self.get_temporal_changes(event_count)

        #print(f"[DEBUG] Received {event_count} events at time {current_time}.")

    def extend_data(self, new_x, new_y, new_t, new_p):
        self.x.extend(new_x)
        self.y.extend(new_y)
        self.t.extend(new_t)
        self.p.extend(new_p)

    def check_y_expansion(self):
        if len(self.y) < 2:
            return False
        y_range = np.ptp(self.y)
        dynamic_threshold = np.mean(self.y) + 1.2 * np.std(self.y)
        #print(f"[DEBUG] Y-axis expansion check: min={min(self.y)}, max={max(self.y)}, range={y_range}, dynamic_threshold={dynamic_threshold}")
        if y_range > dynamic_threshold:
            #print(f"[DEBUG] Y-axis expansion detected at blast #{self.blast_counter} with range {y_range}.")
            return True
        return False

    def check_x_expansion_post_y(self):
        if len(self.x) < 2:
            return False
        x_range = np.ptp(self.x)
        dynamic_x_threshold = np.mean(self.x) + 1.2 * np.std(self.x)
        #print(f"[DEBUG] X-axis expansion check: range={x_range}, threshold={dynamic_x_threshold}")
        if x_range > dynamic_x_threshold:
            #print(f"[DEBUG] X-axis expansion confirmed post Y-expansion at blast #{self.blast_counter} with range {x_range}.")
            return True
        return False

    def check_spatial_coherence(self):
        data = np.column_stack((self.x, self.y))
        if len(data) < 5:
            #print("[DEBUG] Not enough data points for spatial coherence check.")
            return False

        clustering = DBSCAN(eps=15, min_samples=3).fit(data)
        labels = clustering.labels_
        unique_labels = set(labels)
        largest_cluster = max([list(labels).count(i) for i in unique_labels if i != -1], default=0)
        #print(f"[DEBUG] Largest cluster size: {largest_cluster}")
        return largest_cluster > 3


class ROSHistogramVisualizer:
    def __init__(self):
        rospy.init_node('event_histogram_visualizer', anonymous=True)
        self.event_stream = EventStream()
        self.subscriber = rospy.Subscriber('/event_topic', EventArray, self.event_stream.callback)
        self.fig, self.ax = plt.subplots()

    def update_plot(self, frame):
        y_array = np.array(self.event_stream.y)
        if len(y_array) == 0:
            return  # Skip frame if no events

        histogram, bin_edges = np.histogram(y_array, bins=24, range=(0, 480))
        peaks, properties = find_peaks(histogram, height=10)
        peak_positions = bin_edges[peaks] + np.diff(bin_edges)[0] / 2
        #print(f"[DEBUG] Histogram peaks: {peak_positions}")

        if self.event_stream.check_y_expansion():
            if self.event_stream.check_x_expansion_post_y():
                if self.event_stream.check_spatial_coherence():
                    print(f"[ALERT] Blast #{self.event_stream.blast_counter} confirmed with spatial coherence!")

        self.ax.cla()
        self.ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align='edge')
        self.ax.plot(peak_positions, properties['peak_heights'], 'rx')
        plt.draw()

    def run(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        plt.show()


if __name__ == '__main__':
    visualizer = ROSHistogramVisualizer()
    visualizer.run()
