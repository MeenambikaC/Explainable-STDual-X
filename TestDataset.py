class TestDataset(Dataset):
    def __init__(self, eval_data):
        self.eval_data = eval_data
        self.num_sessions = [len(user_sessions) for user_sessions in self.eval_data]  # List of number of sessions for each user
        self.num_seqs = [len(session) for user_sessions in self.eval_data for session in user_sessions]  # Total sequences across all users

    def __len__(self):
        # Total length of dataset will be the sum of all sequences across all users and sessions
        return sum(len(self.eval_data[user_idx][session_idx]) for user_idx in range(len(self.eval_data))
                   for session_idx in range(len(self.eval_data[user_idx])))

    def __getitem__(self, idx):
        # Find the user index and session index dynamically
        cumulative_length = 0
        for user_idx in range(len(self.eval_data)):
            for session_idx in range(len(self.eval_data[user_idx])):
                session_length = len(self.eval_data[user_idx][session_idx])
                if cumulative_length + session_length > idx:
                    seq_idx = idx - cumulative_length
                    data = self.eval_data[user_idx][session_idx][seq_idx]

                    # Debugging statements
                    debug_log(f"Index: {idx}, User Index: {user_idx}, Session Index: {session_idx}, Sequence Index: {seq_idx}")

                    # Check if data is None
                    if data is None:
                        error_log(f"Returned data is None for index: {idx} in testdata")
                    return data, user_idx

                cumulative_length += session_length
        
        # If we get here, idx is out of bounds
        raise IndexError("Index out of bounds for dataset.")