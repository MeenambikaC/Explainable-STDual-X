from math import floor, ceil
class TrainDataset(Dataset):
    def __init__(self, training_data, batch_size):
        self.training_data = training_data
        self.batch_size = batch_size
        flattened_length_train = sum(len(item) if isinstance(item, list) else 1 for sublist in training_data for item in (sublist if isinstance(sublist, list) else [sublist]))
        self.epoch_batch_count = ceil(flattened_length_train / batch_size)
        print(self.epoch_batch_count,"epoch_batch_count")

    def __len__(self):
        
        return self.batch_size * self.epoch_batch_count
        

    def __getitem__(self, idx):
        while True:
            try:
                genuine_user_idx = np.random.randint(0, len(self.training_data))
                imposter_user_idx = np.random.randint(0, len(self.training_data))
                # Ensure imposter_user_idx is different from genuine_user_idx
                while imposter_user_idx == genuine_user_idx:
                    imposter_user_idx = np.random.randint(0, len(self.training_data))
                
                # Validate the lengths of genuine_user and imposter_user data
                if len(self.training_data[genuine_user_idx]) == 0 or len(self.training_data[imposter_user_idx]) == 0:
                    raise ValueError("Empty user data detected.")
                
                genuine_sess_1 = np.random.randint(0, len(self.training_data[genuine_user_idx]))
                genuine_sess_2 = np.random.randint(0, len(self.training_data[genuine_user_idx]))
                
                # Ensure genuine_sess_2 is different from genuine_sess_1
                while genuine_sess_2 == genuine_sess_1:
                    genuine_sess_2 = np.random.randint(0, len(self.training_data[genuine_user_idx]))
                
                # Validate the lengths of genuine_sess_1 and genuine_sess_2 data
                if len(self.training_data[genuine_user_idx][genuine_sess_1]) == 0 or len(self.training_data[genuine_user_idx][genuine_sess_2]) == 0:
                    raise ValueError("Empty session data detected.")
                
                imposter_sess = np.random.randint(0, len(self.training_data[imposter_user_idx]))
                
                # Validate the length of imposter_sess data
                if len(self.training_data[imposter_user_idx][imposter_sess]) == 0:
                    raise ValueError("Empty imposter session data detected.")
                
                genuine_seq_1 = np.random.randint(0, len(self.training_data[genuine_user_idx][genuine_sess_1]))
                genuine_seq_2 = np.random.randint(0, len(self.training_data[genuine_user_idx][genuine_sess_2]))
                imposter_seq = np.random.randint(0, len(self.training_data[imposter_user_idx][imposter_sess]))
#                 print(genuine_user_idx,genuine_sess_1,genuine_sess_2,imposter_user_idx,imposter_sess)
                anchor = self.training_data[genuine_user_idx][genuine_sess_1][genuine_seq_1]
                positive = self.training_data[genuine_user_idx][genuine_sess_2][genuine_seq_2]
                negative = self.training_data[imposter_user_idx][imposter_sess][imposter_seq]


                return anchor, positive, negative, genuine_user_idx, imposter_user_idx
            
            except ValueError as e:
                print_log(f"Encountered ValueError: {str(e)}. Retrying with new indices.")