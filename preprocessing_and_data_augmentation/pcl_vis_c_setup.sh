#!/bin/bash
set -e # exit, if one command fails

# Basics
sudo apt update
sudo apt upgrade -y
sudo apt install -y wget tar git cmake make build-essential

# Dependency for vis_c
sudo apt install -y libfmt-dev

# Dependencies for VTK and PCL
sudo apt install -y libboost-all-dev libeigen3-dev libflann-dev libqhull-dev libopenni-dev libgtest-dev libusb-1.0-0-dev pcaputils pcapfix libpng-dev libpng++-dev mesa-utils freeglut3-dev

# VTK
wget https://www.vtk.org/files/release/9.0/VTK-9.0.1.tar.gz
tar -xf VTK-9.0.1.tar.gz
rm VTK-9.0.1.tar.gz
cd VTK-9.0.1
mkdir build
cd build
cmake ..
make -j 4
sudo make install -j 4
cd ../..

# PCL
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build
cd build
cmake ..
make
sudo make install
