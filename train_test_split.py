import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration parameters
base_path = "/kaggle/working/activity_data"
sequence_length = SEQUENCE_LENGTH   # Length of each IMU sequence
overlap = OVERLAP  # Overlap between sequences
activity_list_train = []
activity_list_test = []
activity_list_val = []

# Set specific users for testing
test_users = ['5.csv','6.csv']
print(f"Selected users {test_users} as test users for all activities")

# Iterate through each activity
for activity_id in activity_list:
    print(f"Processing activity: {activity_id}")
    activity_dir = os.path.join(base_path, f'{activity_id}')

    user_list_train = []
    user_list_test = []
    user_list_val = []

    for user_id in os.listdir(activity_dir):
        print(f"Processing user: {user_id}")
        file_path = os.path.join(activity_dir, user_id)
        imu_data = pd.read_csv(file_path)

        # Apply IMU feature extraction and drop unnecessary columns
        imu_data = apply_imu_feature_extraction(imu_data, imu_prefixes)
        imu_data = imu_data.drop(["subject_id", "activityID"], axis=1)
        imu_data = scaling(imu_data)

        # Generate sequences
        sequence_data = []
        num_samples = len(imu_data)
        num_sequences = (num_samples - sequence_length) // overlap + 1

        for i in range(num_sequences):
            sequence_start = i * overlap
            sequence_end = sequence_start + sequence_length
            if sequence_end <= num_samples:
                sequence = imu_data.iloc[sequence_start:sequence_end].copy()
                sequence_data.append(sequence.to_numpy())

        # Split data according to LOSO-CV with specific test users
        if user_id in test_users:
            # Use all sequences from these users as test data
            print(f"if : {user_id}")
            user_list_test.append(sequence_data)
        else:
             # print(f"else : {user_id}")
            # Split remaining users' data into train and validation sets
            train_data, val_data = train_test_split(sequence_data, test_size=0.2, random_state=12345)
              
            user_list_train.append(train_data)
            user_list_val.append(val_data)

    # Append results to the main lists for each activity
    activity_list_train.append(user_list_train)
    activity_list_test.append(user_list_test)
    activity_list_val.append(user_list_val)

training_imu_data=activity_list_train
testing_imu_data=activity_list_test
validation_imu_data=activity_list_val