#!/usr/bin/env python3
import rospy
import numpy as np
import time
from collections import deque
from gazebo_blast3d.msg import EventArray, BlastBox2D

class PDEBlastDetector:
    """
    A PDE-based event camera blast detector that:
      - Subscribes to /event_topic (gazebo_blast3d::EventArray).
      - Maintains 1D histograms in X and Y directions.
      - Applies a PDE diffusion step for smoothing each update.
      - Performs threshold-based detection on the PDE-smoothed histograms.
      - Simulates a 'ground truth' blast occurrence at specified times, to compute
        precision, recall, F1-score, and accuracy.
      - Measures detection latency (time from receiving the message to deciding 'blast detected').

    Usage:
    1) Launch Gazebo with your .cpp plugin (unchanged).
    2) Run this Python node:  rosrun your_package pde_blast_detector.py
    3) Observe logs for detection metrics.
    """

    def __init__(self):
        # Node init
        rospy.init_node("pde_blast_detector", anonymous=True)

        # Basic camera geometry (match your plugin, default 640x480)
        self.height = rospy.get_param("~height", 480)
        self.width  = rospy.get_param("~width", 640)

        # PDE parameters
        self.D  = rospy.get_param("~diff_coeff", 0.2)  # diffusion coefficient
        self.dt = rospy.get_param("~pde_dt", 1.0)      # PDE time step
        self.dx = rospy.get_param("~pde_dx", 1.0)      # PDE spatial step

        # Detection thresholds
        # e.g., we require consecutive frames of 'blast suspicion' to confirm a blast
        self.verification_threshold = rospy.get_param("~verification_thresh", 2)

        # min_data_threshold => how many events must accumulate before we do anything
        self.min_data_threshold = rospy.get_param("~min_data_threshold", 40)

        # We'll bin X and Y into 20 bins each
        self.bin_count = rospy.get_param("~bin_count", 20)

        # Buffers for storing recent events
        self.x_buffer = deque(maxlen=50000)
        self.y_buffer = deque(maxlen=50000)

        # We'll keep a small history of PDE-smoothed histograms for thresholding
        self.hist_y_history = []
        self.hist_x_history = []

        # Detection logic
        self.consecutive_blast_frames = 0

        # We store detection latencies
        self.latency_measurements = []

        # Confusion matrix counters (for time-based ground truth blasts)
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        # Subscribe to the plugin's event data
        self.sub_events = rospy.Subscriber(
            "/event_topic",
            EventArray,
            self.event_callback,
            queue_size=1
        )

        rospy.loginfo("PDEBlastDetector node started with PDE(D=%.2f, dt=%.2f, dx=%.2f)" %
                      (self.D, self.dt, self.dx))

    def event_callback(self, msg):
        receive_time = time.time()
        # Extract event coordinates and push into x,y buffers
        for evt in msg.events:
            self.x_buffer.append(evt.x)
            self.y_buffer.append(evt.y)

        # If we have enough data, attempt detection
        if len(self.x_buffer) >= self.min_data_threshold:
            self.update_histograms(receive_time)

    def update_histograms(self, receive_time):
        # 1) Build histograms of X and Y from the current buffers
        x_array = np.array(self.x_buffer, dtype=int)
        y_array = np.array(self.y_buffer, dtype=int)

        hist_y, _ = np.histogram(y_array, bins=self.bin_count, range=(0, self.height))
        hist_x, _ = np.histogram(x_array, bins=self.bin_count, range=(0, self.width))

        # 2) Append to our short history
        self.hist_y_history.append(hist_y)
        self.hist_x_history.append(hist_x)
        # Keep only last 5 for thresholding
        if len(self.hist_y_history) > 5:
            self.hist_y_history.pop(0)
        if len(self.hist_x_history) > 5:
            self.hist_x_history.pop(0)

        # 3) PDE smoothing: We take the last 2 histograms
        if len(self.hist_y_history) > 1:
            old_y = self.hist_y_history[-2].astype(float)
            new_y = self.hist_y_history[-1].astype(float)
            smoothed_y = self.apply_pde(old_y, new_y)
            self.hist_y_history[-1] = smoothed_y

        if len(self.hist_x_history) > 1:
            old_x = self.hist_x_history[-2].astype(float)
            new_x = self.hist_x_history[-1].astype(float)
            smoothed_x = self.apply_pde(old_x, new_x)
            self.hist_x_history[-1] = smoothed_x

        # 4) Check if a blast is detected
        is_blast_detected = self.check_blast()

        # 5) If detected, measure latency and log
        if is_blast_detected:
            latency = time.time() - receive_time
            self.latency_measurements.append(latency)
            rospy.loginfo("[ALERT] Blast detected! Latency = %.4f s" % latency)

        # 6) Compare detection with ground truth at this moment
        actual_blast = self.ground_truth_blast()
        self.update_confusion_matrix(is_blast_detected, actual_blast)

    def apply_pde(self, old_hist, new_hist):
        """
        Applies one iteration of the PDE diffusion step:
            h[i] = new_hist[i] + D * (dt / dx^2) * ( new_hist[i+1] - 2*new_hist[i] + new_hist[i-1] )
        """
        h_smoothed = new_hist.copy()
        coeff = self.D * (self.dt / (self.dx**2))
        for i in range(1, len(h_smoothed)-1):
            laplacian = (h_smoothed[i+1] - 2*h_smoothed[i] + h_smoothed[i-1])
            h_smoothed[i] = h_smoothed[i] + coeff*laplacian
        # Ensure no negative values, round to int
        h_smoothed = np.clip(h_smoothed, 0, None)
        return np.round(h_smoothed).astype(int)

    def check_blast(self):
        """
        We check PDE-smoothed histograms for Y & X:
         - compute dynamic threshold: mean + k * std
         - if any bin > threshold for both Y & X => suspicion
         - after consecutive frames => confirm a blast
        """
        if len(self.hist_y_history) == 0 or len(self.hist_x_history) == 0:
            return False

        hist_y = self.hist_y_history[-1]
        hist_x = self.hist_x_history[-1]

        # Gather last up to 5 hist
        window_y = np.array(self.hist_y_history[-5:], dtype=float)
        window_x = np.array(self.hist_x_history[-5:], dtype=float)

        avg_y = np.mean(window_y, axis=0)
        std_y = np.std(window_y, axis=0)
        threshold_y = avg_y + 1.8 * std_y
        y_exceed = (hist_y > threshold_y).any()

        avg_x = np.mean(window_x, axis=0)
        std_x = np.std(window_x, axis=0)
        threshold_x = avg_x + 1.4 * std_x
        x_exceed = (hist_x > threshold_x).any()

        if y_exceed and x_exceed:
            self.consecutive_blast_frames += 1
        else:
            self.consecutive_blast_frames = 0

        if self.consecutive_blast_frames >= self.verification_threshold:
            self.consecutive_blast_frames = 0
            return True
        return False

    #-----------------------------------------------------------------------#
    #                  Simulated Ground Truth & Metrics                     #
    #-----------------------------------------------------------------------#
    def ground_truth_blast(self):
        """
        Example 'ground truth' method. 
        Suppose a real blast starts every 15 seconds and lasts 5 seconds.
        We'll define: a blast is active if (T mod 30) in [5..10].
        Adjust to match your scenario or keep for demonstration.
        """
        t_now = time.time()
        # For a stable simulation reference, you could also use ROS time or something else:
        # t_now = rospy.get_time()  # but that might be 0-based
        cycle = (t_now % 30.0)
        if cycle >= 5.0 and cycle < 10.0:
            return True
        return False

    def update_confusion_matrix(self, is_blast_detected, actual_blast):
        if is_blast_detected and actual_blast:
            self.TP += 1
        elif is_blast_detected and not actual_blast:
            self.FP += 1
        elif (not is_blast_detected) and actual_blast:
            self.FN += 1
        else:
            self.TN += 1

    def precision(self):
        denom = self.TP + self.FP
        return (float(self.TP)/denom) if denom>0 else 0.0

    def recall(self):
        denom = self.TP + self.FN
        return (float(self.TP)/denom) if denom>0 else 0.0

    def accuracy(self):
        total = self.TP + self.FP + self.TN + self.FN
        return (self.TP + self.TN)/float(total) if total>0 else 0.0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p+r)>0:
            return 2.0 * p * r / (p+r)
        return 0.0

    def avg_latency(self):
        return np.mean(self.latency_measurements) if self.latency_measurements else 0.0

if __name__ == "__main__":
    detector = PDEBlastDetector()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    # Print final metrics
    rospy.loginfo("=== Final PDE Blast Detection Metrics ===")
    rospy.loginfo(" Precision: %.3f", detector.precision())
    rospy.loginfo(" Recall:    %.3f", detector.recall())
    rospy.loginfo(" F1-score:  %.3f", detector.f1_score())
    rospy.loginfo(" Accuracy:  %.3f", detector.accuracy())
    rospy.loginfo(" Avg Latency (s):  %.5f", detector.avg_latency())
