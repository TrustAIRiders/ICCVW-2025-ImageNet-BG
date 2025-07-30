#!/bin/bash

if [ -z "$1" ]; then
	  echo "Usage: $0 <path_to_dataset>"
	    exit 1
fi

DATASET_PATH="$1"
cd "$DATASET_PATH" || exit

gdown 1tXdE867qY9WzvHO-eSL8dFJRkWZ8l3Eh -O ImageNet-BG.zip
gdown 1GW-D9iwy9uQIJ6t8JtalrYVvejwruF3U -O ImageNet-BG-extras.zip

unzip ImageNet-BG.zip
unzip ImageNet-BG-extras.zip

rm ImageNet-BG.zip
rm ImageNet-BG-extras.zip

echo "Download and extraction complete."
