import face_recognition
import os
import numpy as np


# Directory where your dataset is located
dataset_folder = "C:/Users/bobur/OneDrive/Рабочий стол/new face id/new face id/faces"  # Replace with the actual path to your dataset

# Initialize lists to store face encodings and names
known_face_encodings = []
known_face_names = []

# Iterate through the dataset folders
for folder_name in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder_name)
    print(folder_path)
    if os.path.isdir(folder_path):  # Check if it's a directory
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]  # Assuming only one face per image
            known_face_encodings.append(face_encoding)
            known_face_names.append(folder_name)

# Save the encodings and names to files
np.save('known_face_encodings.npy', known_face_encodings)
np.save('known_face_names.npy', known_face_names)