# TREX_IA
## Introduction
This part will describe the implementation of deep learning techniques to control the robot through the pathway thanks to the pictures taken by the webcam.

The different objectives are:
- Import the pictures from the storage disk
- Load pictures paths and labels
- Balance data
- Prepare arrays
- Augment and process pictures
- Create and train the model

## Find the notebook
The neural network is designed and trained on [Google Colab](https://colab.research.google.com) This platform provides solutions to create Python notebooks and get much more GPU power to train faster the models.

The TREX notebook can be found at this [address](https://colab.research.google.com/drive/1Inww_IHnbZclx8BwDfj4aP4N3cu4Dl9k?usp=sharing) You can also choose to create a new notebook with File > New Notebook.

The notebook is divided into several code blocks which can be executed independently. At the first run, Colab will connect you to a new session. You can modify the execution type on Execution > Modify execution type to choose GPU or CPU.

## Import the files
To train the neural network model, the first step will be to import the pictures taken from the webcam to the attributed session. 

Colab can be connected to Google Drive easily. It's recommended to create a zip folder containing all the pictures and the log file and upload it somewhere in your Drive.

To mount your Drive use the right method:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
An authorization will be asked to connect Colab to Drive.

Then declare the path to your .zip containing the pictures and the log file:
```python
zip_name = "pictures.zip"
drive_zip_path = '/content/drive/MyDrive/your_path_to_zip_file'
```
To import the .zip into your workspace:
```python
!cp {drive_zip_path} .
!unzip -q {zip_name}
!rm {zip_name}
```

## Load pictures paths and labels
The next step is to load the pictures from the workspace to the python code as a Panda Frame for easier processing.
The .csv log file generated while taking pictures indicates for each row the path (e.g. the timestamp) for each .jpeg file and its associated label. A function is needed to return the image path from the log file:
```python
def getName(filePath):
 # Get image path and folder path
 myImagePathL = filePath.split('/')\[-2:]
 # Join to get complete path
 myImagePath = os.path.join(myImagePathL[0],myImagePathL[1])
 return myImagePath
```
Then the Panda dataframe is created with two columns: the image path and the label corresponding to the steering.
```python
def importDataInfo(path):
 # Create pandas df
 data = pd.DataFrame()
 # Declare columns name
 columns = ['Path','Steering']
 # Read csv file and assign colum name
 dataNew = pd.read_csv(os.path.join(path, f'log.csv'), names = columns)
 print("Number of lines in csv: ", dataNew.shape[0])
 # Print path from first image
 print(print(getName(dataNew['Path'][0])))
 # Get simplified path and replace path column with new path string
 dataNew['Path'] = dataNew['Path'].apply(getName)
 # Append to pd df
 data = data.append(dataNew, True)
 print('Total Images Imported:', data.shape[0])
 return data
```
An overview of the dataframe can be displayed with the pd.head():
```python
data = importDataInfo('pictures')
print(data.head())
```

## Balance data
To complete

## Prepare arrays
To train the neural network, the pictures and the labels needs to be separated into two numpy arrays.
```python
def loadData(path, data):
 # Creating lists of images and steering
 imagesPath = []
 steering = []
 # For each row of data
 for i in range(len(data)):
 # Get row object
 indexed_data = data.iloc[i]
 # Append to list path to the image
 imagesPath.append(os.path.join(path,indexed_data[0].split('/')[1]))
 # Append to list steering value
 steering.append(float(indexed_data[1]))
 # Convert lists to array
 imagesPath = np.asarray(imagesPath)
 steering = np.asarray(steering)
 return imagesPath, steering
```
Loading an image is possible with cv2.imread(). To print it in the console, a modified cv2.imshow() method is provided by Colab:
```python
from google.colab.patches import cv2_imshow
imagesPath, steerings = loadData('testimage', data)
print('No of Path Created for Images ',len(imagesPath),len(steerings))
img = cv2.imread(imagesPath[5])
print(img.shape)
cv2_imshow(img)
```
The picture size can vary from the webcam used in the project.
