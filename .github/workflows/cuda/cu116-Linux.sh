#!/bin/bash

OS=ubuntu1804

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-${OS}-11-6-local_11.6.2-510.47.03-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-11-6-local_11.6.2-510.47.03-1_amd64.deb
sudo apt-key add /var/cuda-repo-${OS}-11-6-local/7fa2af80.pub

sudo apt-get -qq update
sudo apt install cuda cuda-nvcc-11-6 cuda-libraries-dev-11-6
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda-repo-${OS}-11-6-local_11.6.2-510.47.03-1_amd64.deb