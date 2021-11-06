# Master Project
## iOS App
Here you can find the instructions on how to run the iOS app.

1. Clone this repository.
2. Open the `ios_app` directory with Xcode.
3. Connect a compatible Apple device to your machine running Xcode. Note: Since the app uses the LiDAR scanner, it only works on Apple devices with a LiDAR scanner. It is not possible to run this app in the simulator or on an Apple device without a LiDAR scanner.
4. Follow this guide to run the app: [Running Your App in the Simulator or on a Device](https://developer.apple.com/documentation/xcode/running-your-app-in-the-simulator-or-on-a-device)
Note: As mentioned before, running the app in the simulator will not work.
## Preprocessing and Data Augmentation
Here you can find the install instructions for the preprocessing/data augmentation program called ``vis_c`` on Ubuntu.

1. Run ``pcl_vis_c_setup.sh`` to install PCL and ``vis_c`` dependencies.
2. Create a directory ``build`` in the same directory as the files ``main.cpp`` and ``CMakeLists.txt``.
3. Go into the ``build`` directory with ``cd build`` and execute ``cmake ..`` then, execute ``make``.
4. Run ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib``.
5. Execute ``./vis_c`` to check if it is working.

    Optional:

6. To make ``vis_c`` available on the entire machine execute ``sudo cp vis_c /usr/local/bin``.
7. To make step 4 permanent add ``LD_LIBRARY_PATH=/usr/local/lib`` to the file ``/etc/environment`` or extend the entry, if ``LD_LIBRARY_PATH`` is already in there; you need to log out and back in for the change to take effect.

## Neural Network
Here you can find the instructions on how to train and test our neural network.

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
