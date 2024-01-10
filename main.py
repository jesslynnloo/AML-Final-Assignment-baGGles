from post_modelling import load_model, process_video_and_vote
from extract import extract_frames # Import your module for video preprocessing
import torchvision.transforms as transforms
import os
from resize_normalize import process_all_images
import shutil


def main():
    # Specify the path to the video and model
    video_path = 'C:\\Users\\Jesslynn\\Desktop\\Master\\Sem 1\\WOA7015 Advanced Machine Learning\\Assignments\\Final Assignment\\Arm Flapping SSBD (ML Class)\\7.mp4'

    # Get the base directory of the main.py script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up folders within 'input_video' directory
    input_video_dir = os.path.join(base_dir, 'input_video')
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted_frames_folder = os.path.join(input_video_dir, 'frames_extracted')
    processed_frames_folder = os.path.join(input_video_dir, 'processed_frames')

    # Ensure these directories exist
    os.makedirs(extracted_frames_folder, exist_ok=True)
    os.makedirs(processed_frames_folder, exist_ok=True)

    model_path = 'best_CNN-LSTM_model.pth'

    # Define transforms (same as used during training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the model
    model = load_model(model_path)

    # Preprocess the video (extract frames, resize, normalize)
    extract_frames(video_path, os.path.join(extracted_frames_folder, video_base_name), video_base_name)
    process_all_images(extracted_frames_folder, processed_frames_folder)

    # Process the video and get the final prediction
    final_prediction = process_video_and_vote(model, os.path.join(processed_frames_folder, video_base_name), transform)
    print("Final Prediction:", final_prediction)

    if os.path.exists(input_video_dir):
        # Recursively delete the folder and its contents
        shutil.rmtree(input_video_dir)


if __name__ == "__main__":
    main()
