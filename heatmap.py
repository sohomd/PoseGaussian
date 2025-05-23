import cv2
import os
import numpy as np
from tqdm import tqdm
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

# Register all modules (necessary for mmpose to work correctly)
register_all_modules()

# Initialize the model
config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py' # Path to config file downloaded
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth' # Path to checkpointfile downloaded
model = init_model(config_file, checkpoint_file, device='cpu')  # Use 'cuda:0' for GPU

def generate_heatmap_for_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        tqdm.write(f"Failed to load image from {image_path}")
        return None

    results = inference_topdown(model, image)
    if isinstance(results, list):
        for person in results:
            keypoints = person.pred_instances.keypoints
            if keypoints is not None:
                heatmap_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                keypoint_radius = 30
                for keypoint in keypoints[0]:
                    x, y = keypoint
                    if x >= 0 and y >= 0:
                        cv2.circle(heatmap_image, (int(x), int(y)), keypoint_radius, 1, -1)

                for _ in range(3):  # Apply blur multiple times
                    heatmap_image = cv2.GaussianBlur(heatmap_image, (41, 41), 0)

                # Normalize and apply colormap
                heatmap_image = cv2.normalize(heatmap_image, None, 0, 1, cv2.NORM_MINMAX)
                heatmap_image = cv2.applyColorMap(np.uint8(255 * heatmap_image), cv2.COLORMAP_JET)

                return heatmap_image 
    return None

def process_folder(base_folder, output_base_folder):
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    for subfolder in tqdm(subfolders, desc=f"Processing subfolders in {base_folder}"):
        subfolder_path = os.path.join(base_folder, subfolder)
        output_subfolder = os.path.join(output_base_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for image_name in tqdm(image_files, desc=f"{subfolder}", leave=False):
            image_path = os.path.join(subfolder_path, image_name)
            heatmap_image = generate_heatmap_for_image(image_path)

            if heatmap_image is not None:
                heatmap_output_path = os.path.join(output_subfolder, f"{os.path.splitext(image_name)[0]}.png")
                cv2.imwrite(heatmap_output_path, np.uint8(heatmap_image))
            else:
                tqdm.write(f"❌ Failed: {image_name}")

# Paths (update as needed)
train_dir = '' # Your directory for Dataset Training Images train/img
val_dir = '' # Your directory for Dataset Validation Images val/img 
output_folder1 = '/render_data/train/heatmap' #Your directory for Dataset in train folder (subfolder- heatmaps)
output_folder2 = 'render_data/val/heatmap' #Your directory for Dataset in val folder (subfolder- heatmaps)

process_folder(train_dir, output_folder1)
process_folder(val_dir, output_folder2)
