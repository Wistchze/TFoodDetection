# inference.py

import os
import random
import numpy as np
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FoodInferenceConfig(Config):
    NAME = 'foods_cfg'
    NUM_CLASSES = 33
    # Set batch size to 1 for inference
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Disable data augmentation for inference
    DETECTION_MIN_CONFIDENCE = 0.7  # Adjust as needed

# Visualization functions as defined earlier
def display_instances(image, boxes, masks, class_ids, class_names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print('No instances to display.')
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = random_colors(n_instances)

    # Create a blank image for masks
    masked_image = image.copy()

    for i in range(n_instances):
        color_mask = colors[i]

        # Bounding box
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), color_mask, 2)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = f'{label} {score:.2f}' if score else label
        cv2.putText(masked_image, caption, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_mask, 2)

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color_mask, alpha=0.5)

    return masked_image

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N):
    import colorsys
    hsv = [(i / N, 1, 1) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# Define the GUI Application
class ObjectDetectionApp:
    def __init__(self, root, model, class_names):
        self.root = root
        self.model = model
        self.class_names = class_names

        self.root.title('Mask R-CNN Object Detection')

        # Create a frame for buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        # Create a button to upload images
        self.upload_btn = tk.Button(self.button_frame, text='Upload Image', command=self.upload_image)
        self.upload_btn.pack(side='left', padx=10)

        # Create a button to save annotated image
        self.save_btn = tk.Button(self.button_frame, text='Save Annotated Image', command=self.save_image, state='disabled')
        self.save_btn.pack(side='left', padx=10)

        # Create labels to display images
        self.original_label = tk.Label(root, text='Original Image')
        self.original_label.pack(padx=10, pady=5)

        self.original_image_label = tk.Label(root)
        self.original_image_label.pack()

        self.annotated_label = tk.Label(root, text='Annotated Image')
        self.annotated_label.pack(padx=10, pady=5)

        self.annotated_image_label = tk.Label(root)
        self.annotated_image_label.pack()

        # Store the annotated image for saving
        self.annotated_image = None

    def upload_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp'), ('All files', '*.*')]
        )
        if not file_path:
            return

        try:
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror('Error', 'Unable to read the selected image.')
                return

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform detection
            results = self.model.detect([image_rgb], verbose=0)
            r = results[0]

            # Visualize results
            annotated_image = display_instances(image_rgb, r['rois'], r['masks'],
                                               r['class_ids'], self.class_names, r['scores'])

            # Convert images to PIL Image
            original_pil = Image.fromarray(image_rgb.astype(np.uint8))
            annotated_pil = Image.fromarray(annotated_image.astype(np.uint8))

            # Resize images if they're too large
            max_size = 600
            original_pil.thumbnail((max_size, max_size), Image.LANCZOS)
            annotated_pil.thumbnail((max_size, max_size), Image.LANCZOS)

            # Convert to ImageTk
            self.original_imgtk = ImageTk.PhotoImage(image=original_pil)
            self.annotated_imgtk = ImageTk.PhotoImage(image=annotated_pil)

            # Update the labels with the images
            self.original_image_label.configure(image=self.original_imgtk)
            self.original_image_label.image = self.original_imgtk

            self.annotated_image_label.configure(image=self.annotated_imgtk)
            self.annotated_image_label.image = self.annotated_imgtk

            # Store the annotated image for saving
            self.annotated_image = annotated_pil

            # Enable the save button
            self.save_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror('Error', f'An error occurred during detection:\n{e}')

    def save_image(self):
        if self.annotated_image is None:
            messagebox.showwarning('Warning', 'No annotated image to save.')
            return

        # Open file dialog to save the image
        save_path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG files', '*.png'), ('JPEG files', '*.jpg *.jpeg'), ('All files', '*.*')]
        )
        if not save_path:
            return

        try:
            # Save the annotated image
            self.annotated_image.save(save_path)
            messagebox.showinfo('Success', f'Annotated image saved to {save_path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save image:\n{e}')

def main():
    # Define paths
    ROOT_DIR = os.path.abspath('./')
    MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
    TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, 'foods_cfg20241208T1534', 'mask_rcnn_foods_cfg_0060.h5')  # Path to your trained weights

    # Check if trained weights exist
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f'Trained weights not found at {TRAINED_MODEL_PATH}.')
        return

    # Create the inference config
    config = FoodInferenceConfig()
    config.display()

    # Create model in inference mode
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

    # Load weights
    print('Loading weights ', TRAINED_MODEL_PATH)
    model.load_weights(TRAINED_MODEL_PATH, by_name=True)

    # Class names
    class_names = [
        "BG",
        "-",
        "18Friedegg",
        "Ayam",
        "Fried Rice",
        "Lalapan",
        "Sambal",
        "Tumis mie",
        "apple",
        "ayam-kentucky-dada",
        "ayam-kentucky-paha",
        "banana",
        "beef hamburger",
        "chicken-burger",
        "fried tofu",
        "indomie_goreng",
        "nasi_putih",
        "nugget",
        "omelet",
        "orange",
        "paha_ayam_goreng",
        "pisang",
        "rendang sapi",
        "rice",
        "sambal",
        "stir-fried kale",
        "tahu goreng",
        "tahu_goreng",
        "telur_dadar",
        "telur_rebus",
        "tempe goreng",
        "tempe_goreng",
        "tumis_kangkung"
    ]

    # Initialize Tkinter
    root = tk.Tk()
    app = ObjectDetectionApp(root, model, class_names)
    root.mainloop()

if __name__ == '__main__':
    main()
