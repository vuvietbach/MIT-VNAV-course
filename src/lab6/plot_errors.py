#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return None
    try:
        data = np.loadtxt(file_path, skiprows=1)
        if data.size == 0:
            return None
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    log_dir = "/home/bach/projects/VNAV-labs/lab6_ws/src/lab6"
    
    methods = {
        "5-point RANSAC": os.path.join(log_dir, "pose_error_estimator_0_ransac_1.txt"),
        "8-point RANSAC": os.path.join(log_dir, "pose_error_estimator_1_ransac_1.txt"),
        "2-point RANSAC": os.path.join(log_dir, "pose_error_estimator_2_ransac_1.txt"),
        "Arun 3D-3D RANSAC": os.path.join(log_dir, "pose_error_estimator_3_ransac_1.txt"),
        "5-point No RANSAC": os.path.join(log_dir, "pose_error_estimator_0_ransac_0.txt"),
    }
    
    # Load all data
    data_dict = {}
    for name, path in methods.items():
        data = load_data(path)
        if data is not None and data.shape[0] > 0:
            data_dict[name] = data

    if not data_dict:
        print("No valid error log data found.")
        return

    # Plot 1: Rotation Error
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():
        t = data[:, 0] - data[0, 0]  # Align time to start at 0
        rot_err = data[:, 1]
        plt.plot(t, rot_err, label=name, alpha=0.8)
    
    plt.title("Relative Rotation Error Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Rotation Error (deg)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "rotation_errors.png"), dpi=300)
    plt.close()

    # Plot 2: Translation Error (Normalized)
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():
        t = data[:, 0] - data[0, 0]  # Align time to start at 0
        trans_err = data[:, 4]  # Normalized translation error
        plt.plot(t, trans_err, label=name, alpha=0.8)
    
    plt.title("Relative Translation Error Over Time (Normalized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Translation Error (Normalized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "translation_errors.png"), dpi=300)
    plt.close()
    
    print("Plots generated successfully as rotation_errors.png and translation_errors.png.")

    print("\n" + "="*96)
    print(f"{'Method':<25} | {'Avg Rot Error (deg)':<20} | {'Avg Trans Error (Abs)':<20} | {'Avg Trans Error (Norm)':<22}")
    print("-" * 96)
    for name, data in data_dict.items():
        avg_rot = np.mean(data[:, 1])
        avg_trans_abs = np.mean(data[:, 3])
        avg_trans_norm = np.mean(data[:, 4])
        print(f"{name:<25} | {avg_rot:<20.6f} | {avg_trans_abs:<20.6f} | {avg_trans_norm:<22.6f}")
    print("="*96 + "\n")

if __name__ == "__main__":
    main()
