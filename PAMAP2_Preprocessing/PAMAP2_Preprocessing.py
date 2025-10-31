# Load data
list_of_files = ['/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject101.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject102.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject103.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject104.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject105.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject106.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject107.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject108.dat',
                 '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol/subject109.dat' ]

subjectID = [1,2,3,4,5,6,7,8,9]

colNames = ["timestamp", "activityID","heartrate"]
IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro_1', 'handGyro_2', 'handGyro_3', 
           'handMagne_1', 'handMagne_2', 'handMagne_3',
           'handOrientation_1', 'handOrientation_2', 'handOrientation_3', 'handOrientation_4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro_1', 'chestGyro_2', 'chestGyro_3', 
           'chestMagne_1', 'chestMagne_2', 'chestMagne_3',
           'chestOrientation_1', 'chestOrientation_2', 'chestOrientation_3', 'chestOrientation_4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro_1', 'ankleGyro_2', 'ankleGyro_3', 
           'ankleMagne_1', 'ankleMagne_2', 'ankleMagne_3',
           'ankleOrientation_1', 'ankleOrientation_2', 'ankleOrientation_3', 'ankleOrientation_4']

columns = colNames + IMUhand + IMUchest + IMUankle

directory= '/kaggle/input/pampa2/PAMAP2_Dataset/Protocol'

dataCollection = pd.DataFrame()
data_dict={}
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    procData = pd.read_table(file, header=None, sep='\s+')
    procData.columns = columns
    procData['subject_id'] = int(file[-5])
    dataCollection = pd.concat([dataCollection, procData], ignore_index=True)

dataCollection.reset_index(drop=True, inplace=True)

def dataCleaning(dataCollection):
        dataCollection = dataCollection.drop(['handOrientation_1', 'handOrientation_2', 'handOrientation_3', 'handOrientation_4',
                                             'chestOrientation_1', 'chestOrientation_2', 'chestOrientation_3', 'chestOrientation_4',
                                             'ankleOrientation_1', 'ankleOrientation_2', 'ankleOrientation_3', 'ankleOrientation_4','chestTemperature',
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3','ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'handAcc6_1','handAcc6_2','handAcc6_3',
           'heartrate','handTemperature','timestamp'
                                             ],
                                             axis = 1)  # removal of orientation columns as they are not needed
        dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
        dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerce') #removal of non numeric data in cells
        dataCollection = dataCollection.interpolate() #removal of any remaining NaN value cells by constructing new data points in known set of data points
        
        return dataCollection

dataCol = dataCleaning(dataCollection)
dataCol.reset_index(drop = True, inplace = True)

unique_activity_ids = dataCol['activityID'].unique()
unique_subject_ids = dataCol['subject_id'].unique()

training_list = [ 1,  2,  3, 17, 16, 12, 13,  4,  7,  6,  5, 24]
validation_list = [ 1,  2,  3, 17, 16, 12, 13,  4,  7,  6,  5, 24]
testing_list = [ 1,  2,  3, 17, 16, 12, 13,  4,  7,  6,  5, 24]
all_list = [ 1,  2,  3, 17, 16, 12, 13,  4,  7,  6,  5, 24]


# When both hand and chest placements are considered - We need to add a logic here for every placement
imu_prefixes = ['handAcc16', 'handGyro', 'handMagne','chestAcc16','chestGyro','chestMagne','ankleGyro','ankleAcc6','ankleMagne']
