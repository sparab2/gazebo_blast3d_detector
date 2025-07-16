#!/usr/bin/env python3
import rospy
import numpy as np
import time
from collections import deque
from gazebo_blast3d.msg import EventArray

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class PDEBlastDetectorWithPlots:
    """
    PDE-based event camera blast detector that, on shutdown, shows:
      1) Latency distribution (histogram).
      2) Three confusion matrices:
         - Unnormalized raw counts (integer).
         - Row-normalized.
         - Column-normalized.
    Uses a time-based ground truth in [5..10) mod 30s.
    """

    def __init__(self):
        rospy.init_node("pde_blast_detector_with_plots", anonymous=True)

        # Basic geometry
        self.height = rospy.get_param("~height", 480)
        self.width  = rospy.get_param("~width", 640)

        # PDE parameters
        self.D  = rospy.get_param("~diff_coeff", 0.2)
        self.dt = rospy.get_param("~pde_dt", 1.0)
        self.dx = rospy.get_param("~pde_dx", 1.0)

        # Detection thresholds
        self.verification_threshold = rospy.get_param("~verification_thresh", 2)
        self.min_data_threshold = rospy.get_param("~min_data_threshold", 40)
        self.bin_count = rospy.get_param("~bin_count", 20)

        # Buffers for storing events
        self.x_buffer = deque(maxlen=50000)
        self.y_buffer = deque(maxlen=50000)

        # PDE-smoothed history
        self.hist_y_history = []
        self.hist_x_history = []

        self.consecutive_blast_frames = 0
        self.latency_measurements = []

        # Confusion matrix counters
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        # Subscribe to event data
        rospy.Subscriber("/event_topic", EventArray, self.event_callback, queue_size=1)

        rospy.loginfo(
            "PDEBlastDetectorWithPlots started. PDE params: D=%.2f, dt=%.2f, dx=%.2f",
            self.D, self.dt, self.dx
        )

    def event_callback(self, msg):
        receive_time = time.time()
        # Accumulate events
        for evt in msg.events:
            self.x_buffer.append(evt.x)
            self.y_buffer.append(evt.y)

        # If enough data, attempt PDE-based detection
        if len(self.x_buffer) >= self.min_data_threshold:
            self.update_histograms(receive_time)

    def update_histograms(self, receive_time):
        # Build histograms
        x_array = np.array(self.x_buffer, dtype=int)
        y_array = np.array(self.y_buffer, dtype=int)

        hist_y, _ = np.histogram(y_array, bins=self.bin_count, range=(0, self.height))
        hist_x, _ = np.histogram(x_array, bins=self.bin_count, range=(0, self.width))

        self.hist_y_history.append(hist_y)
        self.hist_x_history.append(hist_x)

        # Keep only last 5
        if len(self.hist_y_history) > 5:
            self.hist_y_history.pop(0)
        if len(self.hist_x_history) > 5:
            self.hist_x_history.pop(0)

        # PDE smoothing
        if len(self.hist_y_history) > 1:
            old_y = self.hist_y_history[-2].astype(float)
            new_y = self.hist_y_history[-1].astype(float)
            self.hist_y_history[-1] = self.apply_pde(old_y, new_y)

        if len(self.hist_x_history) > 1:
            old_x = self.hist_x_history[-2].astype(float)
            new_x = self.hist_x_history[-1].astype(float)
            self.hist_x_history[-1] = self.apply_pde(old_x, new_x)

        # Detection
        detected = self.check_blast()
        if detected:
            latency = time.time() - receive_time
            self.latency_measurements.append(latency)
            rospy.loginfo("[ALERT] Blast detected! Latency=%.4f s", latency)

        # Compare with time-based ground truth
        actual = self.ground_truth_blast()
        self.update_confusion_matrix(detected, actual)

    def apply_pde(self, old_hist, new_hist):
        """
        PDE diffusion step:
         h[i] = new_hist[i] + D*(dt/dx^2)*( new_hist[i+1] - 2*new_hist[i] + new_hist[i-1] )
        """
        h_smoothed = new_hist.copy()
        coeff = self.D * (self.dt / (self.dx**2))
        for i in range(1, len(h_smoothed)-1):
            laplacian = (h_smoothed[i+1] - 2*h_smoothed[i] + h_smoothed[i-1])
            h_smoothed[i] += coeff * laplacian
        h_smoothed = np.clip(h_smoothed, 0, None)
        return np.round(h_smoothed).astype(int)

    def check_blast(self):
        if not self.hist_y_history or not self.hist_x_history:
            return False

        hist_y = self.hist_y_history[-1]
        hist_x = self.hist_x_history[-1]

        # Check last up to 5 PDE-smoothed frames
        window_y = np.array(self.hist_y_history[-5:], dtype=float)
        avg_y = np.mean(window_y, axis=0)
        std_y = np.std(window_y, axis=0)
        threshold_y = avg_y + 1.8 * std_y
        y_exceed = (hist_y > threshold_y).any()

        window_x = np.array(self.hist_x_history[-5:], dtype=float)
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

    def ground_truth_blast(self):
        """
        e.g. blast active if (time % 30) in [5..10)
        """
        now = time.time()
        cycle = now % 30.0
        return (5.0 <= cycle < 10.0)

    def update_confusion_matrix(self, detected, actual):
        if detected and actual:
            self.TP += 1
        elif detected and not actual:
            self.FP += 1
        elif not detected and actual:
            self.FN += 1
        else:
            self.TN += 1

    #--- Basic metrics ---
    def precision(self):
        denom = self.TP + self.FP
        return (self.TP / denom) if denom > 0 else 0.0

    def recall(self):
        denom = self.TP + self.FN
        return (self.TP / denom) if denom > 0 else 0.0

    def accuracy(self):
        total = self.TP + self.FP + self.TN + self.FN
        return ((self.TP + self.TN)/float(total)) if total>0 else 0.0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p+r) > 0:
            return 2.0*p*r/(p+r)
        return 0.0

    def avg_latency(self):
        if self.latency_measurements:
            return np.mean(self.latency_measurements)
        return 0.0

    #--- Final plots (latency + 3 confusion matrices) ---
    def show_plots(self):
        """
        Display 4 subplots:
         1) Latency histogram
         2) Unnormalized CM (integer, 'd' format)
         3) Row-normalized CM (floats, '.2f' format)
         4) Column-normalized CM (floats, '.2f' format)
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, axes = plt.subplots(1, 4, figsize=(18,4))

        # 1) Latency distribution
        axes[0].hist(self.latency_measurements, bins=10, color='blue', edgecolor='black')
        axes[0].set_title("Latency Distribution")
        axes[0].set_xlabel("Latency (s)")
        axes[0].set_ylabel("Count")

        # Build raw confusion matrix in typical row=actual, col=pred
        # => [ [TN, FP],
        #      [FN, TP] ]
        cm_raw = np.array([[self.TN, self.FP],
                           [self.FN, self.TP]], dtype=int)

        # 2) Unnormalized: integer data => 'd' format
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_raw,
                                       display_labels=["No Blast", "Blast"])
        disp1.plot(ax=axes[1], values_format='d', colorbar=False)
        axes[1].set_title("CM (Unnormalized)")

        # 3) Row-normalized => each row sums to 1 => float => '.2f' format
        # row_sums => shape (2,1)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid divide-by-zero
        cm_row_norm = cm_raw.astype(float)/row_sums
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_row_norm,
                                       display_labels=["No Blast","Blast"])
        disp2.plot(ax=axes[2], values_format='.2f', colorbar=False)
        axes[2].set_title("CM (Row-Normalized)")

        # 4) Column-normalized => each column sums to 1 => float => '.2f'
        col_sums = cm_raw.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm_col_norm = cm_raw.astype(float)/col_sums
        disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_col_norm,
                                       display_labels=["No Blast","Blast"])
        disp3.plot(ax=axes[3], values_format='.2f', colorbar=False)
        axes[3].set_title("CM (Column-Normalized)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    detector = PDEBlastDetectorWithPlots()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    # On shutdown, print final metrics
    prec = detector.precision()
    rec  = detector.recall()
    f1   = detector.f1_score()
    acc  = detector.accuracy()
    lat  = detector.avg_latency()

    rospy.loginfo("=== Final PDE Blast Detection Metrics ===")
    rospy.loginfo(" Precision:  %.3f", prec)
    rospy.loginfo(" Recall:     %.3f", rec)
    rospy.loginfo(" F1-score:   %.3f", f1)
    rospy.loginfo(" Accuracy:   %.3f", acc)
    rospy.loginfo(" Avg Latency %.5f s", lat)

    # Show the final plots
    detector.show_plots()
