import pickle
import os
import numpy as np
import random
from tensorflow import keras
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Dense



seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tensorflow.random.set_seed(seed_constant)
path = '/content/drive/MyDrive/dataset/UCF50'
all_classes = os.listdir(path)
img_wid = 64
img_hgt= 64
seq_length= 20
Dataset_dir= '/content/drive/MyDrive/dataset/UCF50'
classes_train = all_classes[:20]

classes_train

def feature_extraction(video_path_to):
    frame_list = []
    video_reader = cv2.VideoCapture(video_path_to)
    video_frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_per_frame = max(int(video_frame_count / seq_length), 1)

    for frame_counter in range(seq_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_per_frame)
        ret, frame = video_reader.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (img_wid, img_hgt))
        normalized_frame = resized_frame / 255.0
        frame_list.append(normalized_frame)

    video_reader.release()  # Release video reader
    return frame_list

def prepare_dataset():
    features = []
    labels = []
    video_paths_list = []

    for class_index, class_name in enumerate(classes_train):
        print(f"Extracting data for the class named {class_name}")
        class_file_path = os.listdir(os.path.join(Dataset_dir, class_name))

        for file_name in class_file_path:
            video_path = os.path.join(Dataset_dir, class_name, file_name)
            frames = feature_extraction(video_path)

            if len(frames) == seq_length:
                features.append(np.array(frames))  # Append directly as numpy array
                labels.append(class_index)
                video_paths_list.append(video_path)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels, video_paths_list

import numpy as np
import pickle

Features, labels, paths = prepare_dataset()

# Save data in chunks to avoid memory issues
chunk_size = 100
for i in range(0, len(Features), chunk_size):
    with open(f'/content/drive/MyDrive/ucf_processed/features20_{i//chunk_size}.pkl', 'wb') as f:
        pickle.dump(Features[i:i+chunk_size], f)

with open('/content/drive/MyDrive/ucf_processed/labels20.pkl', 'wb') as f:
    pickle.dump(labels, f)

with open('/content/drive/MyDrive/ucf_processed/video_paths_list20.pkl', 'wb') as f:
    pickle.dump(paths, f)

# Print a message indicating successful save
print("Arrays saved as pickle files to Google Drive.")




with open('/content/drive/MyDrive/ucf_processed/labels20.pkl', 'rb') as f:
    loaded_labels = pickle.load(f)

with open('/content/drive/MyDrive/ucf_processed/video_paths_list20.pkl', 'rb') as f:
    loaded_video_paths_list = pickle.load(f)

# Now you can use loaded_features, loaded_labels, and loaded_video_paths_list in your code


np.array(loaded_labels).shape



# Function to load chunks
def load_chunks(file_prefix, num_chunks):
    loaded_data = []
    for i in range(num_chunks):
        file_path = f'/content/drive/MyDrive/ucf_processed/{file_prefix}_{i}.pkl'
        with open(file_path, 'rb') as f:
            loaded_data.extend(pickle.load(f))
    return np.array(loaded_data)

# Load features chunks
num_feature_chunks = 25  # Adjust this based on the number of chunks you have
X_train = load_chunks('features20', num_feature_chunks)
One_hot_encoded_labels = to_categorical(loaded_labels)


# Print some information
print(f"Train samples: {len(X_train)}")




with open('/content/drive/MyDrive/ucf_processed/features_test20.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('/content/drive/MyDrive/ucf_processed/labels_test20.pkl', 'rb') as f:
    y_test = pickle.load(f)
y_test = to_categorical(y_test)



def create_LRCN_model():
    # Create a Sequential model
    model = Sequential()

    # Convolutional layers applied to each frame in the sequence
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'), input_shape=(seq_length, img_hgt, img_wid, 3)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    # Flatten the output of convolutional layers
    model.add(TimeDistributed(Flatten()))

    # Apply LSTM layer to capture temporal dependencies
    model.add(LSTM(32))

    # Output layer with softmax activation for classification
    model.add(Dense(len(classes_train), activation='softmax'))

    # Display the model summary
    model.summary()

    return model
LRCN_model = create_LRCN_model()

# Display the success message.
print("Model Created Successfully!")


# Create an Instance of Early Stopping Callback.
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min', restore_best_weights = True)

checkpoint_callback = ModelCheckpoint(
    filepath='/content/drive/MyDrive/ucf_model/LRCN201.h5',
    monitor='val_loss',  # You can change this to 'val_accuracy' or another metric
    mode='min',  # Change to 'max' if monitoring accuracy and you want the maximum
    save_best_only=True,  # Save only the best model
    verbose=1
)
# Compile the model and specify loss function, optimizer and metrics to the model.
LRCN_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

# Start training the model.
LRCN_model_training_history = LRCN_model.fit(x = X_train, y = One_hot_encoded_labels, epochs = 70, batch_size = 4 , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback,checkpoint_callback])

import matplotlib.pyplot as plt

def show_graph(model_history, metric_1, metric_2, plot_name):
    value_1 = model_history.history[metric_1]  # Fix: Access history dictionary correctly
    value_2 = model_history.history[metric_2]  # Fix: Access history dictionary correctly

    epochs = range(len(value_1))  # Fix: Use the length of either metric_1 or metric_2
    plt.plot(epochs, value_1, 'blue', label=metric_1)
    plt.plot(epochs, value_2, 'red', label=metric_2)
    plt.title(str(plot_name))
    plt.legend()
    plt.show()  # Fix: Display the plot

# Example usage:
# Assuming you have trained the model and have its history
# Replace 'accuracy' and 'val_accuracy' with the actual metric names you want to plot
show_graph(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Training vs Validation Accuracy')


show_graph(LRCN_model_training_history, 'loss', 'val_loss', 'Training vs Validation loss')


import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
classifir = load_model('/content/drive/MyDrive/ucf_model/LRCN20.h5')

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, f1_score
import numpy as np

# Assuming you have trained your model and obtained predictions on the test set
y_pred = classifir.predict(X_test)  # Replace X_test with your test features
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f'Accuracy: {accuracy:.4f}')

# Precision
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
print(f'Precision: {precision:.4f}')
# F1 Score
f1score = f1_score(y_true_classes, y_pred_classes, average='weighted')
print(f'F1_Score: {f1score:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_true_classes, y_pred_classes)
print('Classification Report:')
print(class_report)


IMAGE_HEIGHT=64
IMAGE_WIDTH=64
CLASSES_LIST = classes_train

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = classifir.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()


classes_train

import numpy as np
# Construct the output video path.
output_video_file_path = f'/content/drive/MyDrive/kinetics_model/jp.mp4'
input_video_file_path ='/content/drive/MyDrive/ucf_testing/JumpRope/v_JumpRope_g01_c01.avi'
# Perform Action Recognition on the Test Video.
predict_on_video(input_video_file_path, output_video_file_path, 20)


