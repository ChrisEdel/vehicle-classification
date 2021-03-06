# Vehicle Classification Using Commercial Off-The-Shelf LiDAR and Camera Sensors (Master's Project 2021)

## Project Summary

The aim of this project is to use the LiDAR sensor of the new Apple devices in combination with the camera sensors in order to classify cars (based on the make and model) using deep learning. We used the iPhone 12 Pro for gathering our data (i.e. labelled 3D scans of cars). Overall, we collected about 400 different scans. Since these scans tend to get quite big (in our case about 500 MB each), preprocessing is necessary. Furthermore, in order increase the size of the dataset, we implemented and used several data augmentation methods. The preprocessed and augmented data is then given as input to our custom neural network, the Color3DNet, which can classify the car makes and even the car models. More information on all these topics can be found below, along with specific instructions on how to run this project.

## Usage Instructions

Here you can find the instructions on how to run the different parts of the projects.

### iOS App
1. Clone this repository.
2. Open the `ios_app` directory with Xcode.
3. Connect a compatible Apple device to your machine running Xcode. Note: Since the app uses the LiDAR scanner, it only works on Apple devices with a LiDAR scanner. It is not possible to run this app in the simulator or on an Apple device without a LiDAR scanner.
4. Follow this guide to run the app: [Running Your App in the Simulator or on a Device](https://developer.apple.com/documentation/xcode/running-your-app-in-the-simulator-or-on-a-device)
Note: As mentioned before, running the app in the simulator will not work.
### Preprocessing and Data Augmentation
Here you can find the install instructions for the preprocessing/data augmentation program called ``vis_c`` on Ubuntu.

1. Run ``pcl_vis_c_setup.sh`` to install PCL and ``vis_c`` dependencies.
2. Create a directory ``build`` in the same directory as the files ``main.cpp`` and ``CMakeLists.txt``.
3. Go into the ``build`` directory with ``cd build`` and execute ``cmake ..`` then, execute ``make``.
4. Run ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib``.
5. Execute ``./vis_c`` to check if it is working.

    Optional:

6. To make ``vis_c`` available on the entire machine execute ``sudo cp vis_c /usr/local/bin``.
7. To make step 4 permanent add ``LD_LIBRARY_PATH=/usr/local/lib`` to the file ``/etc/environment`` or extend the entry, if ``LD_LIBRARY_PATH`` is already in there; you need to log out and back in for the change to take effect.

### Neural Network
1. Clone this repository.
2. If you have access to a GPU and want to use that GPU for training the neural network, you need to have CUDA installed (tested with *CUDA 11.3* (on Linux) and *CUDA 11.2* (on Windows 10)). Please refer to the corresponding installation guide:  
  - [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
  - [Windows](//docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
3. You need to have *Python 3.8 64-bit* installed.
4. Go into the corresponding directory:  
```cd color3Dnet```
5. Install all of the necessary requirements. In order to do that, you can use the following command:  
```pip3 install -r requirements.txt```
6. Run the script:  
```python3 color3Dnet.py [data_path]```
7. For further information, run:  
```python3 color3Dnet.py -h```

The data for training and testing the neural network can be downloaded [here](https://drive.google.com/file/d/1JBKiznmEAJ4bmBXLSUrOOJJwTacLlc63/view?usp=sharing). Note that this data is already preprocessed with:
  - Floor filter: 0.3
  - Distance filter: 0.05
  - Smoothening: 0.01
  - Subsampling: 10 000

## Results
Here you can find the summary of the most important results obtained in both theses.

### Infrastructure
Different options were used in order to examine the corresponding training time of the neural network. The most important results compare the training time of the neural network when using different CPUs and GPUs, as well as the epochs and time needed in order to reach an accuracy of over 90%.

| ![Average runtimes of an epoch with a batch size of 6.](images/cpu_vs_gpu.png) | 
|:--:| 
| *Average runtimes of an epoch with a batch size of 6.* |

| ![Number of epochs needed to reach an accuracy of over 90%.](images/epochs.png) | 
|:--:| 
| *Number of epochs needed to reach an accuracy of over 90%.* |

| ![Training time needed to reach an accuracy of over 90%.](images/time.png) | 
|:--:| 
| *Training time needed to reach an accuracy of over 90%.* |

### Color3DNet
The following graphs show the train and test accuracies of the Color3DNet during training for different configurations. The train/test split used was 80/20 and the batch size used was 20. 

| ![Training with point clouds of size 10 000 classifying car makes.](images/10k_split_0.2_original_color_avg_accuracy_30_epochs.png) | 
|:--:| 
| *Training with point clouds of size 10 000 classifying car makes.* |

| ![Training with point clouds of size 25 000 classifying car makes.](images/25k_split_0.2_BS_20_avg_accuracy_30_epochs.png) | 
|:--:| 
| *Training with point clouds of size 25 000 classifying car makes.* |

| ![Training with point clouds of size 10 000 classifying car models.](images/10k_split_0.2_models_avg_accuracy_30_epochs.png) | 
|:--:| 
| *Training with point clouds of size 10 000 classifying car models.* |

\
More information can be found in the corresponding theses.
