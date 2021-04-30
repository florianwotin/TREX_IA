# TREX_IA
## Introduction
This part will describe the implementation of deep learning techniques to control the robot through the pathway thanks to the pictures taken by the webcam.

The different objectives are:
- Import the pictures from the storage disk
- Load pictures paths and labels
- Balance data
- Augment and process pictures
- Create and train the model

## Find the notebook
The neural network can be designed and trained on [Google Colab](https://colab.research.google.com) This platform provides solutions to create Python notebooks and get much more GPU power to train faster the models.

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
To import the .zip into your work session:
```python
!cp {drive_zip_path} .
!unzip -q {zip_name}
!rm {zip_name}
```
