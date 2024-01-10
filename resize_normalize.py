import cv2
import os

def resize_and_normalize_image(image_path, output_path, size=(224, 224)):

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image / 255.0
    cv2.imwrite(output_path, normalized_image * 255)

def process_all_images(source_root, destination_root):

    for folder_name in os.listdir(source_root):
        folder_path = os.path.join(source_root, folder_name)
        if os.path.isdir(folder_path):
            # Create corresponding folder in the destination root
            dest_folder_path = os.path.join(destination_root, folder_name)
            os.makedirs(dest_folder_path, exist_ok=True)

            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    # Create corresponding subfolder in the destination folder
                    dest_subfolder_path = os.path.join(dest_folder_path, subfolder_name)
                    os.makedirs(dest_subfolder_path, exist_ok=True)

                    for image_name in os.listdir(subfolder_path):
                        if image_name.lower().endswith('.jpg'):
                            image_path = os.path.join(subfolder_path, image_name)
                            output_path = os.path.join(dest_subfolder_path, image_name)
                            resize_and_normalize_image(image_path, output_path)

# Example usage
# source_root = 'C:\\Users\\Jesslynn\\Desktop\\Master\\Sem 1\\WOA7015 Advanced Machine Learning\\Assignments\\Final Assignment\\Preprocessing\\Frames_extracted'  # Replace with your source folder path
# destination_root = 'C:\\Users\\Jesslynn\\Desktop\\Master\\Sem 1\\WOA7015 Advanced Machine Learning\\Assignments\\Final Assignment\\Preprocessing\\Processed_Frames'  # Replace with your destination folder path
# process_all_images(source_root, destination_root)
