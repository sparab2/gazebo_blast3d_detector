#!/usr/bin/env python3
import rospy
import numpy as np
from gazebo_blast3d.msg import EventArray
from collections import deque
import time
import os
import statistics  # for mean, stdev

class MeasureExecutionTime:
    """
    Measures execution time of a code block and appends it to a file.
    Used to track how long histogram updates or other steps take.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'a') as file:
            file.write(f"{execution_time}\n")

class EventStream:
    """
    Subscribes to an EventArray topic, attempts to detect blasts using
    histogram thresholds, measures:
     - Latency
     - Confusion Matrix (tp, fp, tn, fn) => accuracy, precision, recall, f1
    """
    def __init__(
        self, 
        max_length=1000, 
        height=480, 
        width=640, 
        blast_threshold=8, 
        verification_threshold=5, 
        min_data_threshold=40
    ):
        # Buffers for events
        self.x = deque(maxlen=max_length)
        self.y = deque(maxlen=max_length)
        self.height = height
        self.width = width

        # Histories (lists of histograms)
        self.event_histories_y = []
        self.event_histories_x = []

        # Thresholds & detection logic
        self.blast_threshold = blast_threshold
        self.verification_threshold = verification_threshold
        self.min_data_threshold = min_data_threshold
        self.consecutive_detections = 0

        # Latency
        self.latency_measurements = []

        # Confusion matrix counters
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        # Initialize the ROS subscriber
        # (We won't call rospy.init_node() here; we'll do it in main())
        rospy.Subscriber('/event_topic', EventArray, self.callback)

    def ground_truth_blast(self):
        """
        Placeholder for ground-truth check. Adjust to your scenario:
         - Possibly read a known schedule, or
         - Subscribe to a separate ROS topic containing GT info.
        """
        current_time = time.time() % 60
        return 10 <= current_time < 15

    def callback(self, data):
        receive_time = time.time()
        # Collect events into x,y buffers
        for event in data.events:
            self.x.append(event.x)
            self.y.append(event.y)
        # Then do detection logic
        self.update_histogram(receive_time)

    def update_histogram(self, receive_time):
        """
        Builds histograms of X, Y event coordinates and checks thresholds
        to see if a 'blast' is detected.
        """
        with MeasureExecutionTime('/tmp/blast3d_processing_times.txt'):
            is_blast_detected = False

            if len(self.y) >= self.min_data_threshold:
                y_array = np.array(self.y)
                x_array = np.array(self.x)

                # Build histograms
                histogram_y, _ = np.histogram(
                    y_array, bins=20, 
                    range=(min(y_array), max(y_array))
                )
                histogram_x, _ = np.histogram(
                    x_array, bins=20, 
                    range=(min(x_array), max(x_array))
                )

                self.event_histories_y.append(histogram_y)
                self.event_histories_x.append(histogram_x)

                # Optional smoothing step
                self.apply_diffusion()

                # Check both X & Y conditions
                if self.confirm_blast_y(self.event_histories_y[-1]) and \
                   self.confirm_blast_x(self.event_histories_x[-1]):
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0

                # If we exceed some threshold of consecutive frames, declare a detection
                if self.consecutive_detections >= self.verification_threshold:
                    # Mark detection
                    lat = time.time() - receive_time
                    self.latency_measurements.append(lat)

                    rospy.loginfo(f"[ALERT] Blast detected! Latency: {lat:.3f} s")
                    is_blast_detected = True
                    self.consecutive_detections = 0

            # Compare detection vs ground truth
            actual_blast = self.ground_truth_blast()
            # Update confusion matrix
            if is_blast_detected and actual_blast:
                self.tp += 1
            elif is_blast_detected and not actual_blast:
                self.fp += 1
            elif not is_blast_detected and actual_blast:
                self.fn += 1
            else:
                self.tn += 1

    def apply_diffusion(self):
        """
        Simple smoothing: current_hist = 0.8*prev + 0.2*curr
        """
        if len(self.event_histories_y) > 1:
            self.event_histories_y[-1] = (
                0.8 * self.event_histories_y[-2] + 
                0.2 * self.event_histories_y[-1]
            ).astype(int)

            self.event_histories_x[-1] = (
                0.8 * self.event_histories_x[-2] + 
                0.2 * self.event_histories_x[-1]
            ).astype(int)

    def confirm_blast_y(self, histogram_y):
        window_size = 5
        # If fewer than 5 frames exist, just use them all
        recent = np.array(self.event_histories_y[-window_size:])
        avg_y = np.mean(recent, axis=0)
        std_y = np.std(recent, axis=0)
        threshold_y = avg_y + 1.8 * std_y
        return np.any(histogram_y > threshold_y)

    def confirm_blast_x(self, histogram_x):
        window_size = 5
        recent = np.array(self.event_histories_x[-window_size:])
        avg_x = np.mean(recent, axis=0)
        std_x = np.std(recent, axis=0)
        threshold_x = avg_x + 1.4 * std_x
        return np.any(histogram_x > threshold_x)

    # -------------------------
    # Classification Metrics
    # -------------------------
    def calculate_average_latency(self):
        if self.latency_measurements:
            return sum(self.latency_measurements) / len(self.latency_measurements)
        return 0.0

    def accuracy(self):
        total = self.tp + self.fp + self.tn + self.fn
        return 0.0 if total == 0 else (self.tp + self.tn) / total

    def precision(self):
        denom = self.tp + self.fp
        return 0.0 if denom == 0 else self.tp / denom

    def recall(self):
        denom = self.tp + self.fn
        return 0.0 if denom == 0 else self.tp / denom

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p + r) == 0:
            return 0.0
        return 2.0 * (p * r) / (p + r)

    def reset_state(self):
        """
        Clears all counters, hist buffers, etc. 
        Used for multiple-run scenarios, so each run starts fresh.
        """
        self.x.clear()
        self.y.clear()
        self.event_histories_y.clear()
        self.event_histories_x.clear()
        self.consecutive_detections = 0
        self.latency_measurements.clear()
        self.tp = self.fp = self.tn = self.fn = 0


def main():
    rospy.init_node('blast_event_stream_multirun', anonymous=True)

    # Number of runs (scenarios)
    N_runs = 5  # example: do 5 repeated runs
    run_duration_sec = 15.0  # each run is 15 seconds

    # Prepare arrays for final summary
    all_acc = []
    all_prec = []
    all_rec = []
    all_f1 = []
    all_lat = []

    event_stream = EventStream()  # Our detection object

    for run_id in range(N_runs):
        # 1) Reset state for this run
        event_stream.reset_state()

        # 2) Let the system collect data for 'run_duration_sec'
        rospy.loginfo(f"[Run {run_id+1}/{N_runs}] Starting data collection for {run_duration_sec}s...")
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < run_duration_sec and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # 3) After time expires, compute metrics for this run
        avg_lat = event_stream.calculate_average_latency()
        acc = event_stream.accuracy()
        prec = event_stream.precision()
        rec = event_stream.recall()
        f1 = event_stream.f1_score()

        # Log them to console
        rospy.loginfo(f"[Run {run_id+1}] Metrics:")
        rospy.loginfo(f"  Average Latency: {avg_lat:.3f}s")
        rospy.loginfo(f"  Accuracy:        {acc:.3f}")
        rospy.loginfo(f"  Precision:       {prec:.3f}")
        rospy.loginfo(f"  Recall:          {rec:.3f}")
        rospy.loginfo(f"  F1 Score:        {f1:.3f}\n")

        # 4) Store in lists for final averaging
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_lat.append(avg_lat)

    # 5) After all runs are done, compute mean ± std. dev.
    mean_acc = statistics.mean(all_acc)
    std_acc  = statistics.pstdev(all_acc)  # or statistics.stdev if you want sample stdev
    mean_prec = statistics.mean(all_prec)
    std_prec  = statistics.pstdev(all_prec)
    mean_rec  = statistics.mean(all_rec)
    std_rec   = statistics.pstdev(all_rec)
    mean_f1   = statistics.mean(all_f1)
    std_f1    = statistics.pstdev(all_f1)
    mean_lat  = statistics.mean(all_lat)
    std_lat   = statistics.pstdev(all_lat)

    rospy.loginfo("=== Final multi-run summary ===")
    rospy.loginfo(f" Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    rospy.loginfo(f" Precision: {mean_prec:.3f} ± {std_prec:.3f}")
    rospy.loginfo(f" Recall: {mean_rec:.3f} ± {std_rec:.3f}")
    rospy.loginfo(f" F1: {mean_f1:.3f} ± {std_f1:.3f}")
    rospy.loginfo(f" Avg Latency: {mean_lat:.3f} ± {std_lat:.3f}\n")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
