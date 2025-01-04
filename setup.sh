#!/bin/bash

# Update the package lists and upgrade the system
sudo apt-get update -y
sudo apt update -y
sudo apt upgrade -y

# Install necessary packages
sudo apt install -y git ffmpeg imx500-all libcap-dev python3-opencv python3-picamera2

# Clone the Git repository
if [ ! -d "cam" ]; then
    git clone https://github.com/sethderusha/cam.git
else
    echo "Directory 'cam' already exists. Skipping git clone."
fi

# Reboot the system
read -p "The system needs to reboot to apply changes. Reboot now? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    sudo reboot
else
    echo "Reboot skipped. Please reboot the system manually later."
fi
