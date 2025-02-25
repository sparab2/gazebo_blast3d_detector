import rospy
import numpy as np
from gazebo_blast3d.msg import EventArray
from collections import deque
import time
import os

class MeasureExecutionTime:
    def __init__(self, file_path):
        self.file_path = file_path
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'a') as file:
            file.write(f"{execution_time}\n")

class EventStream:
    def __init__(
        self, 
        max_length=1000, 
        height=480, 
        width=640, 
        blast_threshold=8, 
        verification_threshold=2, 
        min_data_threshold=40
    ):
        # Buffers for events
        self.x = deque(maxlen=max_length)
        self.y = deque(maxlen=max_length)
        self.height = height
        self.width = width

        # Histories
        self.event_histories_y = []
        self.event_histories_x = []

        # Thresholds
        self.blast_threshold = blast_threshold
        self.verification_threshold = verification_threshold
        self.min_data_threshold = min_data_threshold
        self.consecutive_detections = 0

        # Latency
        self.latency_measurements = []
        self.last_receive_time = None

        # Confusion matrix counters
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        rospy.init_node('event_stream')
        rospy.Subscriber('/event_topic', EventArray, self.callback)

    def ground_truth_blast(self):
        """
        Placeholder function to determine whether a real blast is 
        happening at the current time. 

        In a real scenario, this might:
          - Query a known schedule of blasts
          - Check a separate ROS topic for 'true' blast labels
          - Use any other mechanism that provides ground-truth info
        """
        # Example: Suppose we say a blast is always happening between
        # 10 <= time < 15 seconds in each minute, purely as a placeholder
        # current_time = time.time() % 60  # mod 60 for "each minute" logic
        return (current_time >= 10 and current_time < 15)

    def callback(self, data):
        # Called every time we receive an EventArray message
        receive_time = time.time()
        for event in data.events:
            self.x.append(event.x)
            self.y.append(event.y)
        self.update_histogram(receive_time)

    def reset_state(self):
        self.x.clear()
        self.y.clear()
        self.event_histories_y.clear()
        self.event_histories_x.clear()
        self.consecutive_detections = 0
        self.latency_measurements.clear()

        # Reset confusion matrix
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_histogram(self, receive_time):
        # Measure processing time
        with MeasureExecutionTime(
            '/home.md2/sparab2/wind/uncc_wind_control/ros_image/ros_ws/src/gazebo_blast3d_detector/processing_times.txt'
        ):
            is_blast_detected = False

            if len(self.y) > self.min_data_threshold:
                y_array = np.array(self.y)
                x_array = np.array(self.x)

                # Create histograms
                histogram_y, _ = np.histogram(
                    y_array, bins=20, 
                    range=(min(self.y), max(self.y))
                )
                histogram_x, _ = np.histogram(
                    x_array, bins=20, 
                    range=(min(self.x), max(self.x))
                )

                self.event_histories_y.append(histogram_y)
                self.event_histories_x.append(histogram_x)

                self.apply_diffusion()

                # Check Y & X hist thresholds
                if (self.confirm_blast_y(self.event_histories_y[-1]) and
                    self.confirm_blast_x(self.event_histories_x[-1])):

                    self.consecutive_detections += 1
                    if self.consecutive_detections >= self.verification_threshold:
                        # We declare a blast detection
                        current_time = time.time()
                        latency = current_time - receive_time
                        self.latency_measurements.append(latency)

                        print(f"[ALERT] Blast detected! Latency: {latency:.3f} s")

                        is_blast_detected = True
                        self.consecutive_detections = 0
                    else:
                        self.consecutive_detections = 0
            
            # 1) Compare is_blast_detected to ground-truth
            actual_blast = self.ground_truth_blast()

            # 2) Update confusion-matrix counters
            if is_blast_detected and actual_blast:
                self.tp += 1
            elif is_blast_detected and not actual_blast:
                self.fp += 1
            elif not is_blast_detected and actual_blast:
                self.fn += 1
            else:
                self.tn += 1

    def apply_diffusion(self):
        if len(self.event_histories_y) > 1:
            self.event_histories_y[-1] = (
                self.event_histories_y[-2] * 0.8 + 
                self.event_histories_y[-1] * 0.2
            ).astype(int)

            self.event_histories_x[-1] = (
                self.event_histories_x[-2] * 0.8 + 
                self.event_histories_x[-1] * 0.2
            ).astype(int)

    def confirm_blast_y(self, histogram_y):
        recent_history_y = np.array(self.event_histories_y[-5:])
        avg_events_y = np.mean(recent_history_y, axis=0)
        std_dev_y = np.std(recent_history_y, axis=0)
        threshold_y = avg_events_y + 1.8 * std_dev_y
        return np.any(histogram_y > threshold_y)

    def confirm_blast_x(self, histogram_x):
        recent_history_x = np.array(self.event_histories_x[-5:])
        avg_events_x = np.mean(recent_history_x, axis=0)
        std_dev_x = np.std(recent_history_x, axis=0)
        threshold_x = avg_events_x + 1.4 * std_dev_x
        return np.any(histogram_x > threshold_x)

    def calculate_average_latency(self):
        if self.latency_measurements:
            return sum(self.latency_measurements) / len(self.latency_measurements)
        return 0.0

    # -------------------------
    # Classification Metrics
    # -------------------------
    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    def precision(self):
        denom = (self.tp + self.fp)
        if denom == 0:
            return 0.0
        return self.tp / denom

    def recall(self):
        denom = (self.tp + self.fn)
        if denom == 0:
            return 0.0
        return self.tp / denom

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p + r) == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

if __name__ == '__main__':
    try:
        event_stream = EventStream()

        # (Optional) Instead of 100 loops, you might run indefinitely
        for j in range(10):
            # Spin a bit to process incoming events
            rospy.spin()

            # Print Latency + Accuracy stats
            avg_latency = event_stream.calculate_average_latency()
            acc = event_stream.accuracy()
            prec = event_stream.precision()
            rec = event_stream.recall()
            f1 = event_stream.f1_score()

            print(f"\nRun {j+1}:")
            print(f"  Average Latency: {avg_latency:.3f} s")
            print(f"  Accuracy:        {acc:.3f}")
            print(f"  Precision:       {prec:.3f}")
            print(f"  Recall:          {rec:.3f}")
            print(f"  F1 Score:        {f1:.3f}\n")

            # Reset after each run or experiment
            event_stream.reset_state()

    except rospy.ROSInterruptException:
        pass
