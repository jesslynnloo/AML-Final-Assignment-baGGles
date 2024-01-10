import cv2
import os

def augment_image(image, augmentation_type, angle=None):

    if augmentation_type == 'flip':
        return cv2.flip(image, 1)  # Horizontal flip
    elif augmentation_type == 'rotate':
        if angle is not None:
            # Rotate by specified angle
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (width, height))
        return image
    else:
        return image

def augment_bunch(bunch_folder, output_folder, augmentation_type, angle=None):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(bunch_folder):
        if image_name.lower().endswith('.jpg'):
            image_path = os.path.join(bunch_folder, image_name)
            image = cv2.imread(image_path)
            augmented_image = augment_image(image, augmentation_type, angle)
            cv2.imwrite(os.path.join(output_folder, image_name), augmented_image)

def augment_class(class_folder, augmentation_type, angles):

    for bunch_name in os.listdir(class_folder):
        bunch_folder = os.path.join(class_folder, bunch_name)
        for angle in angles:
            output_folder_suffix = f'_{augmentation_type}_{angle}' if augmentation_type == 'rotate' else f'_{augmentation_type}'
            output_folder = os.path.join(class_folder, f'{bunch_name}{output_folder_suffix}')
            augment_bunch(bunch_folder, output_folder, augmentation_type, angle)

base_path = 'C:\\Users\\Jesslynn\\Desktop\\Master\\Sem 1\\WOA7015 Advanced Machine Learning\\Assignments\\Final Assignment\\Data\\train'
rotation_angles = [45, 90]

# # Augment 'Arm flapping'
# augment_class(os.path.join(base_path, 'Arm flapping'), 'flip', [None])
# augment_class(os.path.join(base_path, 'Arm flapping'), 'rotate', rotation_angles)

# Augment 'Finger flicking'
augment_class(os.path.join(base_path, 'Finger flicking'), 'flip', [None])
augment_class(os.path.join(base_path, 'Finger flicking'), 'rotate', rotation_angles)
