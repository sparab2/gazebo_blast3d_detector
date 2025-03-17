#!/usr/bin/env python3

import rospy
import numpy as np
import time
from collections import deque
from gazebo_blast3d.msg import EventArray

class PDEEventDetector:
    def __init__(
        self,
        max_length=1000,
        height=480,
        width=640,
        verification_threshold=2,
        min_data_threshold=40,
        D=0.2,    # PDE diffusion coefficient
        dt=1.0,   # PDE time step
        dx=1.0    # PDE spatial step
    ):
        """
        PDE-based event blast detector.
        Subscribes to /event_topic (EventArray).
        """
        self.height = height
        self.width = width
        self.verification_threshold = verification_threshold
        self.min_data_threshold = min_data_threshold

        # PDE parameters
        self.D = D
        self.dt = dt
        self.dx = dx

        # Deques to store incoming event coords
        self.x = deque(maxlen=max_length)
        self.y = deque(maxlen=max_length)

        # We'll store the last few histogram arrays for threshold logic
        self.histories_y = []
        self.histories_x = []

        # Detection logic
        self.consecutive_detections = 0

        # Latency logging
        self.latency_measurements = []

        # Initialize ROS node
        rospy.init_node("pde_event_detector", anonymous=True)
        rospy.Subscriber("/event_topic", EventArray, self.callback)

        rospy.loginfo("PDEEventDetector node started.")

    def callback(self, msg):
        """
        Called whenever we get an EventArray message.
        """
        receive_time = time.time()
        for ev in msg.events:
            self.x.append(ev.x)
            self.y.append(ev.y)
        self.update_histograms(receive_time)

    def update_histograms(self, receive_time):
        """
        Create histograms, apply PDE smoothing, detect blasts.
        """
        # Only proceed if we have enough data
        if len(self.y) < self.min_data_threshold:
            return

        y_vals = np.array(self.y, dtype=int)
        x_vals = np.array(self.x, dtype=int)

        # Build histograms in a fixed number of bins
        hist_y, _ = np.histogram(y_vals, bins=20, range=(0, self.height))
        hist_x, _ = np.histogram(x_vals, bins=20, range=(0, self.width))

        # Append to histories
        self.histories_y.append(hist_y)
        self.histories_x.append(hist_x)

        # Apply PDE to the current histogram if there's a previous one
        if len(self.histories_y) > 1:
            old_y = self.histories_y[-2].astype(float)
            new_y = self.histories_y[-1].astype(float)

            # PDE step for y-dimension
            y_smoothed = self.apply_pde(old_y, new_y)
            self.histories_y[-1] = y_smoothed

        if len(self.histories_x) > 1:
            old_x = self.histories_x[-2].astype(float)
            new_x = self.histories_x[-1].astype(float)

            # PDE step for x-dimension
            x_smoothed = self.apply_pde(old_x, new_x)
            self.histories_x[-1] = x_smoothed

        # Now check for blasts
        is_blast = self.check_blast()
        if is_blast:
            current_time = time.time()
            latency = current_time - receive_time
            self.latency_measurements.append(latency)
            rospy.loginfo(f"[ALERT] Blast detected! Latency: {latency:.4f} s")

    def apply_pde(self, old_hist, new_hist):
        """
        PDE-based smoothing:
        We do a single finite-difference iteration:
            h_new[i] = new_hist[i] + D*(dt/dx^2)*( new_hist[i+1] - 2*new_hist[i] + new_hist[i-1] )
        We'll also incorporate some blend with the old_hist if you like.
        """
        # We can either do an explicit PDE step on the *new* histogram,
        # or combine old + new in some ratio. Below is a simple approach:
        h = new_hist.copy()

        # PDE coefficient
        coeff = self.D * (self.dt / (self.dx ** 2))

        # We'll ignore boundary changes (i=0, i=len-1) to keep it simple
        for i in range(1, len(h) - 1):
            laplacian = (h[i+1] - 2*h[i] + h[i-1])
            h[i] = h[i] + coeff * laplacian

        # Round back to int
        return np.round(h).astype(int)

    def check_blast(self):
        """
        Simple threshold-based check:
          - We look at the last (say) 1 or 2 PDE-smoothed histograms for y, x
          - If y-axis distribution is above threshold & x-axis distribution is above threshold, 
            we confirm a blast after a certain # consecutive checks.
        """
        if len(self.histories_y) < 1 or len(self.histories_x) < 1:
            return False

        hist_y = self.histories_y[-1]
        hist_x = self.histories_x[-1]

        # Compare to average of last few frames
        recent_y = np.array(self.histories_y[-5:])  # up to 5
        avg_y = np.mean(recent_y, axis=0)
        std_y = np.std(recent_y, axis=0)
        threshold_y = avg_y + 1.8 * std_y
        y_exceed = np.any(hist_y > threshold_y)

        recent_x = np.array(self.histories_x[-5:])
        avg_x = np.mean(recent_x, axis=0)
        std_x = np.std(recent_x, axis=0)
        threshold_x = avg_x + 1.4 * std_x
        x_exceed = np.any(hist_x > threshold_x)

        # If both exceed, then we might have a blast
        if y_exceed and x_exceed:
            self.consecutive_detections += 1
            if self.consecutive_detections >= self.verification_threshold:
                self.consecutive_detections = 0
                return True
            else:
                return False
        else:
            self.consecutive_detections = 0
            return False

    def calculate_average_latency(self):
        if self.latency_measurements:
            return sum(self.latency_measurements) / len(self.latency_measurements)
        return 0.0


if __name__ == "__main__":
    detector = PDEEventDetector()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    avg_latency = detector.calculate_average_latency()
    rospy.loginfo(f"Average detection latency: {avg_latency:.4f} s")
