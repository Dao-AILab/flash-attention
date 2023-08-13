#!/bin/bash

# Strip the periods from the version number
OS_VERSION=$(echo $(lsb_release -sr) | tr -d .)
OS=ubuntu${OS_VERSION}

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-${OS}-12-0-local_12.0.0-525.60.13-1_amd64.deb

sudo dpkg -i cuda-repo-${OS}-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-${OS}-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get -qq update
sudo apt install cuda cuda-nvcc-12-0 cuda-libraries-dev-12-0
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-${OS}-12-0-local_12.0.0-525.60.13-1_amd64.deb
