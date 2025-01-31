import pandas as pd

def load_data(detections_file, ground_truth_file):
    # Load detection and ground truth data
    detections = pd.read_csv(detections_file)
    ground_truth = pd.read_csv(ground_truth_file)
    return detections, ground_truth

def preprocess_data(detections, ground_truth):
    # Reset indices and sort by 'Time' for both dataframes
    detections.reset_index(drop=True, inplace=True)
    ground_truth.reset_index(drop=True, inplace=True)
    detections.sort_values('Time', inplace=True)
    ground_truth.sort_values('Time', inplace=True)
    return detections, ground_truth

def calculate_accuracy(detections, ground_truth, window=2):
    # Print min and max times to understand the range and scale
    print("Detection Times:", detections['Time'].min(), detections['Time'].max())
    print("Ground Truth Times:", ground_truth['Time'].min(), ground_truth['Time'].max())

    matches = 0
    for index, gt_row in ground_truth.iterrows():
        time_lower = gt_row['Time'] - window
        time_upper = gt_row['Time'] + window
        possible_matches = detections[(detections['Time'] >= time_lower) & (detections['Time'] <= time_upper)]

        # Debug output
        if possible_matches.empty:
            print(f"No matches found for ground truth at time {gt_row['Time']}")
        else:
            print(f"Matches found for ground truth at time {gt_row['Time']}")

        for _, det_row in possible_matches.iterrows():
            if det_row['X'] == gt_row['X'] and det_row['Y'] == gt_row['Y']:
                matches += 1
                break

    accuracy = matches / len(ground_truth) if ground_truth.shape[0] > 0 else 0
    return accuracy

def adjust_detection_times(detections, start_time):
    # Subtract start_time from each detection timestamp to normalize
    detections['Time'] = detections['Time'] - start_time
    return detections

if __name__ == "__main__":

    # File paths
    detections_file = '/home.md2/sparab2/wind/uncc_wind_control/ros_image/ros_ws/src/gazebo_blast3d/datasets/detection.csv'
    ground_truth_file = '/home.md2/sparab2/wind/uncc_wind_control/ros_image/ros_ws/src/gazebo_blast3d/datasets/ground_truth_data.csv'

    # Load and preprocess data
    detections, ground_truth = load_data(detections_file, ground_truth_file)

    # Assume start_time is the Unix timestamp at which your detections start
    start_time = detections['Time'].min()  # Getting the earliest detection time

    # Adjust detection times
    detections = adjust_detection_times(detections, start_time)

    detections, ground_truth = preprocess_data(detections, ground_truth)

    # Calculate accuracy
    accuracy = calculate_accuracy(detections, ground_truth)
    print("Accuracy after adjustment:", accuracy)