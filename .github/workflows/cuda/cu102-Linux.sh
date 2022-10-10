#!/bin/bash

OS=ubuntu1804

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-${OS}-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub

sudo apt-get -qq update
sudo apt install cuda cuda-nvcc-10-2 cuda-libraries-dev-10-2
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-${OS}-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb