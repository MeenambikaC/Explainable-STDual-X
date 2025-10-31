def imu_feature_extract(imu_type_data):
    if imu_type_data.shape[0] < 3:  # Minimum required rows for np.gradient with edge_order=2
        print_log("Not enough rows to calculate gradient. Skipping this file.")
        return pd.DataFrame()
    imu_type_data["fft_x"] = np.abs(np.fft.fft(imu_type_data["x"].values))
    imu_type_data["fft_y"] = np.abs(np.fft.fft(imu_type_data["y"].values))
    imu_type_data["fft_z"] = np.abs(np.fft.fft(imu_type_data["z"].values))

    imu_type_data["fd_x"] = np.gradient(imu_type_data["x"].values, edge_order=2)
    imu_type_data["fd_y"] = np.gradient(imu_type_data["y"].values, edge_order=2)
    imu_type_data["fd_z"] = np.gradient(imu_type_data["z"].values, edge_order=2)

    imu_type_data["sd_x"] = np.gradient(imu_type_data["fd_x"].values, edge_order=2)
    imu_type_data["sd_y"] = np.gradient(imu_type_data["fd_y"].values, edge_order=2)
    imu_type_data["sd_z"] = np.gradient(imu_type_data["fd_z"].values, edge_order=2)

    return imu_type_data


def apply_imu_feature_extraction(dataCol, imu_cols):
    new_columns = []  

    for col_prefix in imu_cols:

        imu_data = dataCol[[f"{col_prefix}_1", f"{col_prefix}_2", f"{col_prefix}_3"]].copy()
        imu_data.columns = ['x', 'y', 'z']
        
        extracted_features = imu_feature_extract(imu_data)
        
        new_cols = pd.DataFrame()
        for feature in extracted_features.columns:
            new_cols[f"{col_prefix}_{feature}"] = extracted_features[feature]

        new_columns.append(new_cols)

    dataCol = pd.concat([dataCol] + new_columns, axis=1)

    for col_prefix in imu_cols:
        dataCol.drop(columns=[f"{col_prefix}_1", f"{col_prefix}_2", f"{col_prefix}_3"], inplace=True)
    
    return dataCol

def scaling(dataframe):
    std_scaler = StandardScaler()
    columns_names = list(dataframe.columns)
    dataframe = std_scaler.fit_transform(dataframe.to_numpy())
    dataframe = pd.DataFrame(dataframe, columns=columns_names)
    return dataframe

base_dir = './activity_data'
activity_counts = []

train_dir = os.path.join('./', 'train')
test_dir = os.path.join('./', 'test')
validation_dir = os.path.join('./', 'validation')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

for activity_id in unique_activity_ids:
    
    activity_dir = os.path.join(base_dir, f'{activity_id}')
    os.makedirs(activity_dir, exist_ok=True)
    
    
    saved_csv_files = []
    
    
    for subject_id in unique_subject_ids:
        
        df_activity_subject = dataCol[(dataCol['activityID'] == activity_id) & (dataCol['subject_id'] == subject_id)]
        
        
        row_count = len(df_activity_subject)
        
        if row_count <=1:
            
            print_log(f'Skipped {subject_id} for activity {activity_id} due to insufficient data ({row_count} rows)')
            continue
        
        
        filename = os.path.join(activity_dir, f'{subject_id}.csv')
        
        
        df_activity_subject.to_csv(filename, index=False)
        
        
        activity_counts.append((activity_id, subject_id, row_count))
        
        
        saved_csv_files.append(filename)
        
        print_log(f'Saved {filename}')
    
    
    existing_csv_files = os.listdir(activity_dir)
    current_csv_count = len(existing_csv_files)
    print_log(existing_csv_files,"existing_csv_files")


train_sequences=[]
validation_sequences=[]
test_sequences=[]
activity_list=all_list

