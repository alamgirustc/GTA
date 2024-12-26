import numpy as np

# Load the .npz file
file_path = 'mscoco/feature/tranfifty_feats/1000_tranfifty.npz'  # Update this with the correct path to your file
npz_file = np.load(file_path)

# Access the 'transformed_features' array
transformed_features = npz_file['transformed_features']

# Define the output text file path
output_file_path = 'output_transformed_features.txt'  # Update this if needed

# Define the feature titles for the original features
feature_titles = [
    "x", "y", "w", "h", "Aspect Ratio", "Area", "Perimeter", "Centroid_x", "Centroid_y", "Compactness"
]

# Define suffixes for transformations
transform_suffixes = [' (Original)', ' (Translated)', ' (Rotated)', ' (Scaled)', ' (Sheared)']

# Create the full set of titles for each transformation
full_titles = []
for suffix in transform_suffixes:
    full_titles.extend([f"{title}{suffix}" for title in feature_titles])

# Save the values to the text file
with open(output_file_path, 'w') as f:
    for idx, roi_features in enumerate(transformed_features):
        f.write(f"RoI {idx + 1}:\n")
        # Write the values in labeled format
        for i in range(0, 50, 10):
            labeled_features = [f"{full_titles[j]}: {roi_features[j]:.4f}" for j in range(i, i + 10)]
            f.write("\t".join(labeled_features) + "\n")
        f.write("\n")  # Newline for better readability
