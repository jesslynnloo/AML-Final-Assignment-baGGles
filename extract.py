import cv2
import os


def extract_frames(video_path, output_folder, video_base_name, fps=15, bunch_size=15):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / fps))
    bunch_count = 0
    frame_list = []

    success, frame = cap.read()
    frame_number = 0

    while success:
        if frame_number % frame_interval == 0:
            frame_list.append(frame)
            if len(frame_list) == bunch_size:
                bunch_name = f'video_{video_base_name}_bunch_{bunch_count:03d}'
                bunch_folder = os.path.join(output_folder, bunch_name)
                os.makedirs(bunch_folder, exist_ok=True)
                for i, frame in enumerate(frame_list):
                    cv2.imwrite(os.path.join(bunch_folder, f'frame_{i:03d}.jpg'), frame)
                bunch_count += 1
                frame_list = []

        success, frame = cap.read()
        frame_number += 1

    cap.release()



def process_all_videos_in_folder(videos_folder_path, frames_extracted_folder_path):

    for filename in os.listdir(videos_folder_path):
        if filename.lower().endswith(('.mp4')):  # Check for MP4 files
            video_path = os.path.join(videos_folder_path, filename)
            video_base_name = os.path.splitext(filename)[0]  # Extract base name of the video
            output_folder = os.path.join(frames_extracted_folder_path, video_base_name)
            extract_frames(video_path, output_folder, video_base_name)

# Example usage
# videos_folder_path = 'C:\\Users\\Jesslynn\\Desktop\\Master\\Sem 1\\WOA7015 Advanced Machine Learning\\Assignments\\Final Assignment\\Arm Flapping SSBD (ML Class)'
# frames_extracted_folder_path = 'C:\\Users\\Jesslynn\\Desktop\\Master\\Sem 1\\WOA7015 Advanced Machine Learning\\Assignments\\Final Assignment\\Preprocessing\\Frames_extracted'
# process_all_videos_in_folder(videos_folder_path, frames_extracted_folder_path)
