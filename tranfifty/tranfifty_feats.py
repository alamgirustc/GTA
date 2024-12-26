import numpy as np
import os
import cv2
from tqdm import tqdm

# Set your paths
bbox_folder = r'D:\LSTMTran\mscoco\feature\up_down_100_box'
output_folder = r'D:\LSTMTran\mscoco\feature\tranfifty_feats'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Geometric Transformation Functions
def translate_bbox(bbox, tx, ty):
    bbox[:, 0] += tx
    bbox[:, 1] += ty
    return bbox

def rotate_bbox(bbox, angle):
    for i in range(bbox.shape[0]):
        x_center = bbox[i, 0] + bbox[i, 2] / 2
        y_center = bbox[i, 1] + bbox[i, 3] / 2
        M = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)
        coords = np.hstack((bbox[i, :2], np.ones((1,))))
        rotated_coords = M @ coords.T
        bbox[i, :2] = rotated_coords[:2].T
    return bbox

def scale_bbox(bbox, sx, sy):
    bbox[:, 2] *= sx
    bbox[:, 3] *= sy
    return bbox

def shear_bbox(bbox, shx, shy):
    shear_matrix = np.array([[1, shx, 0], [shy, 1, 0]])
    for i in range(bbox.shape[0]):
        coords = np.hstack((bbox[i, :2], np.ones((1,))))
        sheared_coords = shear_matrix @ coords.T
        bbox[i, :2] = sheared_coords[:2].T
    return bbox

# Geometric features extraction and transformation application for each RoI
def extract_geometric_features_full(bbox):
    """Extracts all the geometric features: x, y, w, h, aspect_ratio, area, perimeter, centroid_x, centroid_y, and compactness."""
    x, y, w, h = bbox
    aspect_ratio = w / h if h != 0 else 0
    area = w * h
    perimeter = 2 * (w + h)
    centroid_x = x + w / 2
    centroid_y = y + h / 2
    compactness = area / (perimeter ** 2) if perimeter != 0 else 0
    return np.array([x, y, w, h, aspect_ratio, area, perimeter, centroid_x, centroid_y, compactness])

# Normalization function
def normalize_features(features):
    max_vals = np.max(features, axis=0)
    min_vals = np.min(features, axis=0)
    # Avoid division by zero
    return (features - min_vals) / (max_vals - min_vals)

# Function to apply transformations and extract geometric features
def extract_transformed_features_full(bboxes, transformations):
    """Applies transformations to the bounding boxes and extracts their geometric features."""
    transformed_features = []

    # For each transformation, apply and then extract the full 10 geometric features
    for transform in transformations:
        transformed_bbox = transform(bboxes.copy())  # Apply transformation
        geo_features = np.zeros((transformed_bbox.shape[0], 10))  # Placeholder for 10 features per bbox

        for i in range(transformed_bbox.shape[0]):
            bbox = transformed_bbox[i]
            if bbox[2] > 0 and bbox[3] > 0:  # Ensure valid width and height
                geo_features[i] = extract_geometric_features_full(bbox)

        # Add the transformed features to the list
        transformed_features.append(geo_features)

    # Concatenate all transformed features horizontally (axis=1) for each RoI
    return np.concatenate(transformed_features, axis=1)

# Function to process all files and ensure 50 features per bounding box
def process_files_full(bbox_folder, output_folder):
    files = [f for f in os.listdir(bbox_folder) if f.endswith('.npy')]

    for file_name in tqdm(files, desc="Processing files", unit="file"):
        try:
            # Load bounding box data
            bbox_path = os.path.join(bbox_folder, file_name)
            bboxes = np.load(bbox_path)

            # Define transformations
            transformations = [
                lambda b: translate_bbox(b, tx=10, ty=15),
                lambda b: rotate_bbox(b, angle=30),  # Rotation center calculated per bbox
                lambda b: scale_bbox(b, sx=1.2, sy=1.2),
                lambda b: shear_bbox(b, shx=0.1, shy=0.1)
            ]

            # Extract and normalize transformed features
            original_features = np.array([extract_geometric_features_full(b) for b in bboxes])  # Extract original 10 features
            transformed_features = extract_transformed_features_full(bboxes, transformations)  # 40 transformed features

            # Concatenate original features (10) and transformed features (40)
            final_features = np.hstack((original_features, transformed_features))

            # Normalize the final features
            final_features = normalize_features(final_features)

            # Save final features (50 features per RoI)
            if final_features.size > 0:
                save_path = os.path.join(output_folder, file_name.replace('.npy', '_tranfifty.npz'))
                np.savez_compressed(save_path, transformed_features=final_features)
            else:
                print(f"No features generated for {file_name}. Skipping saving.")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Execute the processing
process_files_full(bbox_folder, output_folder)
