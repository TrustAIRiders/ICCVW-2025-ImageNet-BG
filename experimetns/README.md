
# ImageNet-BG

## Overview

This project is focused on evaluating the performance of various deep learning models on the ImageNet-BG dataset. The repository contains scripts for generating and downloading model results, running experiments, and using the ImageNet-BG dataset with PyTorch.

## Contents

- `A_generate_model_results.py`: Python Script to generate DataFrame pickle files with logits for the ImageNet-BG dataset using the following models. Here is the code in PyTorch on how to use ImageNet-BG.
```bash
python A_generate_model_results.py
```

- `A_download_model_results.sh`: Shell script to download pre-calculated deep learning model results without having to generate them locally. Run this script as an alternative to A_generate_model_results.py
```bash
sh A_download_model_results.sh
```


- `B_experiments.py`: Python Script to replicate the experiments from the main paper and appendix.
```bash
python B_experiments.py
```

- `imagenet_bg.py`: Python code containing the `ImageNetBG` class for easy use of the ImageNet-BG dataset with PyTorch.

- `imagenet_class_index.json`: JSON file containing the ImageNet class names and IDs.

- `requirements.txt`: Text file containing the list of required Python packages to run the scripts.

## Setup

Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Using ImageNet-BG with PyTorch

To use the ImageNet-BG dataset with PyTorch, utilize the `ImageNetBG` class provided in `imagenet_bg.py`. Below is an example of how to use it:

```python
from imagenet_bg import ImageNetBG

dataset = ImageNetBG(root_dir='path_to_imagenet_bg', labels_file='path_to_imagenet_class_index.json', transform=None)
```

Refer to the code in the repository for more detailed usage examples.

