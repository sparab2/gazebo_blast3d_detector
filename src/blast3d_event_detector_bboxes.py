#!/usr/bin/env python3
import rospy
import numpy as np
from gazebo_blast3d.msg import EventArray, BlastBox2D
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
        elapsed = time.time() - self.start_time
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'a') as f:
            f.write(f"{elapsed}\n")

class EventStreamWithBBoxes:
    def __init__(
        self,
        max_length=1000,
        height=480,
        width=640,
        blast_threshold=10,
        verification_threshold=9,
        min_data_threshold=40
    ):
        # Buffers
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

        # Metrics
        self.latency_measurements = []
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        # IoU
        self.gt_box = None
        self.detected_box = None
        self.iou_scores = []

        # CSV
        self.csv_path = "/tmp/blast3d_metrics_bboxes.csv"
        self.iou_file = "/tmp/blast3d_iou_per_detection.csv"
        self._init_csv()

        rospy.init_node("blast_event_stream_with_bboxes", anonymous=True)
        rospy.Subscriber("/event_topic", EventArray, self.callback_events)
        rospy.Subscriber("/ground_truth_box", BlastBox2D, self.callback_gt_box)

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, 'w') as f:
            f.write("run,avg_latency,accuracy,precision,recall,f1,mean_iou\n")
        os.makedirs(os.path.dirname(self.iou_file), exist_ok=True)
        with open(self.iou_file, 'w') as f:
            f.write("timestamp,iou,det_xmin,det_ymin,det_xmax,det_ymax,gt_xmin,gt_ymin,gt_xmax,gt_ymax\n")

    def callback_events(self, data):
        receive_time = time.time()
        for evt in data.events:
            self.x.append(evt.x)
            self.y.append(evt.y)
        self.update_histogram(receive_time)

    def callback_gt_box(self, msg):
        self.gt_box = (msg.u_min, msg.v_min, msg.u_max, msg.v_max)

    def update_histogram(self, receive_time):
        with MeasureExecutionTime("/tmp/blast3d_processing_times.txt"):
            is_blast_detected = False

            # Only proceed if we have enough events
            if len(self.y) >= self.min_data_threshold:
                y_arr = np.array(self.y)
                x_arr = np.array(self.x)

                # Build histograms
                hist_y, _ = np.histogram(y_arr, bins=20, range=(y_arr.min(), y_arr.max()))
                hist_x, _ = np.histogram(x_arr, bins=20, range=(x_arr.min(), x_arr.max()))
                self.event_histories_y.append(hist_y)
                self.event_histories_x.append(hist_x)
                self.apply_diffusion()

                # Check threshold logic
                if self.confirm_blast_y(hist_y) and self.confirm_blast_x(hist_x):
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0

                # If consecutive frames exceed threshold => possible detection
                if self.consecutive_detections >= self.verification_threshold:
                    lat = time.time() - receive_time

                    # If latency < 0.001, silently skip it. Do not set is_blast_detected.
                    if lat >= 0.001:
                        # This detection is valid
                        self.latency_measurements.append(lat)
                        rospy.loginfo(f"[ALERT] Blast detected! Latency: {lat:.3f}s")
                        is_blast_detected = True

                    # Whether we skip or accept, reset
                    self.consecutive_detections = 0

            # Compare detection to ground truth
            actual_blast = self.is_blast_active()  # or any ground truth logic

            # Update confusion matrix as normal
            if is_blast_detected and actual_blast:
                self.tp += 1
            elif is_blast_detected and not actual_blast:
                self.fp += 1
            elif not is_blast_detected and actual_blast:
                self.fn += 1
            else:
                self.tn += 1



            # if detection + bounding box => compute IoU
            if is_blast_detected and actual_blast and self.detected_box is not None:
                iou_value = self.compute_iou(self.detected_box, self.gt_box)
                self.iou_scores.append(iou_value)
                rospy.loginfo(f"Detected vs. GT box IoU = {iou_value:.3f}")
                now_stamp = time.time()
                with open(self.iou_file, 'a') as f:
                    f.write(f"{now_stamp:.3f},{iou_value:.3f},"
                            f"{self.detected_box[0]},{self.detected_box[1]},"
                            f"{self.detected_box[2]},{self.detected_box[3]},"
                            f"{self.gt_box[0]},{self.gt_box[1]},"
                            f"{self.gt_box[2]},{self.gt_box[3]}\n")

    def is_blast_active(self):
        if self.gt_box is None: return False
        (u1, v1, u2, v2) = self.gt_box
        return (u2 > u1) and (v2 > v1)

    def apply_diffusion(self):
        if len(self.event_histories_y) > 1:
            self.event_histories_y[-1] = (
                0.8*self.event_histories_y[-2] + 0.2*self.event_histories_y[-1]
            ).astype(int)
            self.event_histories_x[-1] = (
                0.8*self.event_histories_x[-2] + 0.2*self.event_histories_x[-1]
            ).astype(int)

    def confirm_blast_y(self, hist_y):
        window_size = 8
        recent = np.array(self.event_histories_y[-window_size:])
        avg_y = np.mean(recent, axis=0)
        std_y = np.std(recent, axis=0)
        thresh_y = avg_y + 1.8 * std_y
        return np.any(hist_y > thresh_y)

    def confirm_blast_x(self, hist_x):
        window_size = 8
        recent = np.array(self.event_histories_x[-window_size:])
        avg_x = np.mean(recent, axis=0)
        std_x = np.std(recent, axis=0)
        thresh_x = avg_x + 1.4 * std_x
        return np.any(hist_x > thresh_x)

    def compute_iou(self, boxA, boxB):
        Ax1, Ay1, Ax2, Ay2 = boxA
        Bx1, By1, Bx2, By2 = boxB
        interX1 = max(Ax1, Bx1)
        interY1 = max(Ay1, By1)
        interX2 = min(Ax2, Bx2)
        interY2 = min(Ay2, By2)
        interW = max(0.0, interX2 - interX1)
        interH = max(0.0, interY2 - interY1)
        interArea = interW * interH
        areaA = max(0.0, Ax2 - Ax1)*max(0.0, Ay2 - Ay1)
        areaB = max(0.0, Bx2 - Bx1)*max(0.0, By2 - By1)
        unionArea = areaA + areaB - interArea
        if unionArea <= 0:
            return 0.0
        return interArea / unionArea

    # metrics
    def calculate_average_latency(self):
        if not self.latency_measurements:
            return 0.0
        return sum(self.latency_measurements)/len(self.latency_measurements)

    def accuracy(self):
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn)/total if total>0 else 0.0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp/denom if denom>0 else 0.0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp/denom if denom>0 else 0.0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p+r)==0: return 0.0
        return 2.0*p*r/(p+r)

    def mean_iou(self):
        if not self.iou_scores:
            return 0.0
        return sum(self.iou_scores)/len(self.iou_scores)

    def reset_state(self):
        self.x.clear()
        self.y.clear()
        self.event_histories_y.clear()
        self.event_histories_x.clear()
        self.consecutive_detections = 0
        self.latency_measurements.clear()
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.gt_box = None
        self.detected_box = None
        self.iou_scores.clear()

    def log_metrics_to_csv(self, run_idx):
        avg_lat = self.calculate_average_latency()
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1_score()
        miou = self.mean_iou()
        with open(self.csv_path, 'a') as f:
            f.write(f"{run_idx},{avg_lat:.3f},{acc:.3f},{prec:.3f},{rec:.3f},{f1:.3f},{miou:.3f}\n")

def main():
    node = EventStreamWithBBoxes()
    r = rospy.Rate(0.1)  # every 10s
    run_idx = 0

    while not rospy.is_shutdown():
        r.sleep()
        run_idx+=1
        avg_lat = node.calculate_average_latency()
        acc = node.accuracy()
        prec = node.precision()
        rec = node.recall()
        f1  = node.f1_score()
        miou = node.mean_iou()

        rospy.loginfo(f"\n=== Stats after run {run_idx} ===")
        rospy.loginfo(f"  Average Latency: {avg_lat:.3f}")
        rospy.loginfo(f"  Accuracy:        {acc:.3f}")
        rospy.loginfo(f"  Precision:       {prec:.3f}")
        rospy.loginfo(f"  Recall:          {rec:.3f}")
        rospy.loginfo(f"  F1 Score:        {f1:.3f}")
        rospy.loginfo(f"  Mean IoU:        {miou:.3f}\n")

        node.log_metrics_to_csv(run_idx)

if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
