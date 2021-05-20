# TREX_IA
## Introduction
This part will describe the implementation of deep learning techniques and image classification to control the robot through a pathway thanks to the pictures taken by the webcam.

The different objectives are:
- Import the files from the storage disk
- Import pictures paths and labels
- Load pictures and labels
- Augment and process pictures
- Create and train the model
- Print the results and save the model

## Find the notebook
The neural network is designed and trained on [Google Colab](https://colab.research.google.com) This platform provides solutions to create Python notebooks and we can get much more GPU power to train faster the models.

The TREX notebook can be found [here](https://colab.research.google.com/drive/1Inww_IHnbZclx8BwDfj4aP4N3cu4Dl9k?usp=sharing). You can also choose to create a new notebook with File > New Notebook.

The notebook is divided into several code blocks which can be executed independently. At the first run, Colab will connect you to a new session. You can modify the execution type on Execution > Modify execution type to choose GPU or CPU.

## Import the files
To train the neural network model, the first step will be to import the pictures taken from the webcam to the attributed session. 

Colab can be connected to Google Drive easily. It's recommended to create a zip folder containing all the pictures and the log file and upload it somewhere in your Drive.

To mount your Drive use the right method:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
An authorization will be asked to connect Colab to Google Drive.

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
The next step is to load the pictures from the workspace as a Panda Frame dataset for easier processing.

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

## Prepare arrays
To train the neural network, the pictures and the labels need to be separated into two numpy arrays.

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

## Pictures augmenting and process
What characterize a good model is what type of data is implemented in it for training. In addition to good labelling, the polyvalence of the pictures dataset will help the model to predict a better direction of steering for the robot. For example, an image taken from the webcam be blurred, flipped or zoomed representing every situation in the real life (weather or luminosity changing, slopes...). Training a model with only one type of pictures taken in the same conditions can affect the prediction accuracy.

The first step is to augment each image of the dataset. Each image will be randomly (with a probability of 50%) rotated, scaled, brightened or flipped:
```python
def augmentImage(imgPath,steering):
	# Function to modify an image and its steering. Dependinga random value, an
	# image can be modified many times

	# Read image from path and convert it as a numpy array
	img =  mpimg.imread(imgPath)

	# Rotation of image
	if np.random.rand() < 0.5:
		pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
		img = pan.augment_image(img)

	# Scaling image
	if np.random.rand() < 0.5:
		zoom = iaa.Affine(scale=(1, 1.2))
		img = zoom.augment_image(img)

	# Multiply all pixels of image to make darker or brighter
	if np.random.rand() < 0.5:
 		brightness = iaa.Multiply((0.5, 1.2))
 		img = brightness.augment_image(img)

	 # Flip image
	 if np.random.rand() < 0.5:
		 img = cv2.flip(img, 1)
		 steering = -steering
 	return img, steering
```
Then each image will be processed to change the color coding and size:
```python
def preProcess(img):
 # Change size and and color coding for an image
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 img = cv2.GaussianBlur(img,  (3, 3), 0)
 img = cv2.resize(img, (320, 240))
 # Change values of pixels from 0-255 to 0-1
 img = img.astype(np.float32) / 255
 return img
```
The size of the image can be changed depending the speed of image computing by the neural network.
You can choose to augment and process each image of the dataset:
```python
imagesAugmentedList = []
steeringList = []

for image, steering in zip(imagesPath, steerings):
	imageAugmented, newSteering = augmentImage(image, steering)
	imagesAugmentedList.append(imageAugmented)
	steeringList.append(steering)
```
```python
imagesProcessedList = []

for image in imagesAugmentedList:
	imageProcessed = preProcess(image)
	imagesProcessedList.append(imageProcessed)
```
You can plot as an example an image with the method plt.imshow().

## Create and train the model
The architecture chosen for the model is a Convolutional Neural Network (CNN). It has the advantage to have better results with images as it is directly inspired of the organization of animal visual cortex. A CNN is adapted for our application of image classification.

The Keras API will be used to create the CNN. All the documentation can be found as this [address](https://keras.io/api/).

The first step is to create a Sequential model defined as a linear stack of layers which will contain our neural network:
```python
model = Sequential()
```
The next step is to add the convolutions layers:
```python
model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=(240, 320, 3), activation='elu'))
model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
model.add(Convolution2D(64, (3, 3), activation='elu'))
model.add(Convolution2D(64, (3, 3), activation='elu'))
```
The number of convolution layers and the parameters depends of the level of details to be analysed in each image.
For each layer:
- The filters represents the dimensionality of the output space or the convolution channels. They are used to extract features from images in the process of convolution. The first convolution layer as 24 filters and the number is increasing for each layer to get deeper details of an image.
- The kernel size represents the convolution window. A little kernel size will increase the precision of features analysis.
- The strides parameter (here (2,2)) determine how much the window shifts by in each of the dimensions (height and width).
- The input shape indicated to the convolution layer the size of the input image. This parameter is only necessary for the first layer.
- The activation function determines the value of the output. 'ELU' is a function that tend to converge cost to zero faster and produce more accurate results.

The second step is to add a Flatten layer to reshape the input as a one-dimension vector:
```python
model.add(Flatten())
```
Then we can add regular densely-connected NN layers with the output:
```python
model.add(Dense(100, activation = 'elu'))
model.add(Dense(50, activation = 'elu'))
model.add(Dense(10, activation = 'elu'))
model.add(Dense(1))
```
The model is compiled with a learning rate of 0.0001 and computes the mean squared error between labels and predictions:
```python
model.compile(Adam(lr=0.0001),loss='mse')
```
The model architecture can be changed to improve performances.

Before training the model, the last step is to split the dataset between the training and validation set:
```python
imagesProcessedArray = np.asarray(imagesProcessedList)
steeringsArray = np.asarray(steeringList)

# Split images into train and test set
xTrain, xVal, yTrain, yVal = train_test_split(imagesProcessedArray, steeringsArray, test_size=0.2,random_state=10)
```
The validation set will be used to compare with the predicted results to compute the performance indicators. You can choose the proportion of the validation set with test_size or the shuffling with random_state.

Finally, you can train the model with model.fit() using the training and validation sets:
```python
history = model.fit(xTrain, yTrain, epochs=100, validation_data=(xVal, yVal))
```
You can choose the number of epochs defining the number of time the model is trained with the entire training set. The results are saved in the history variable.

To get a prediction from a trained model, you can use the method model.predict() with an image converted as a numpy array:
```python
steering = int(model.predict(img))
```

The model can be saved to use it in the robot:
```python
model.save("model.h5")
```

## Alternative way to train the model
Instead of training the model with the entire dataset, we can choose to pick randomly pictures to enhance the versatility of the model. Each image picked will be processed and augmented to fit the model:
```python
def dataGen(imagesPath, steeringList, batchSize, trainFlag):
 # Data generation fitting of the model
 while True:
 # Creating lists of images and steerings batch
 imgBatch = []
 steeringBatch = []

 for i in range(batchSize):
 # Choosing an image randomly
 index = random.randint(0, len(imagesPath) - 1)
 # If using dataGen for training, augmenting image
 if trainFlag:
	img, steering = augmentImage(imagesPath[index], steeringList[index])
 # Else just read the image get steering value
 else:
 	img = mpimg.imread(imagesPath[index])
 	steering = steeringList[index]

 # Then preprocess and append to batch
 img = preProcess(img)
 imgBatch.append(img)
 steeringBatch.append(steering)
 yield (np.asarray(imgBatch),np.asarray(steeringBatch))
 ```
 Then you can call the function during the fitting:
 ```python
 history = model.fit(dataGen(xTrain, yTrain, 100, 1), steps_per_epoch=10, epochs=10, validation_data=dataGen(xVal, yVal, 50, 0), validation_steps=50)