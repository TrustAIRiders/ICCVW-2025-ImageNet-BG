# [ICCVW-2025] ImageNet-BG: A Toolkit and Dataset for Evaluating Vision Model Robustness Against Background Variations

ImageNet-BG is a benchmark and toolkit to evaluate how well vision models handle background changes. 
It transfers foreground objects from ImageNet-S onto diverse backgrounds-realistic (e.g., SUN397, DTD) and 
abstract (e.g., white, black, noise). Our results show even state-of-the-art models are sensitive to background 
context, revealing important vulnerabilities.

## Quick Demo

To run a quick evaluation using a ResNet model:
```
pip install -r ./experiments/requirements.txt

mkdir ImageNet-Background
bash download.sh ./ImageNet-Background/

python ./demo_resnet.py
```

## Project Structure

```
experiments/   - Code to reproduce tables and figures from the paper
toolkit/       - Tools to build custom background variants
demo_resnet.py - Example script to benchmark ResNet
download.sh    - Script to fetch ImageNet-BG backgrounds
```

**Dataset:** Download ImageNet-BG from [Google Drive](https://drive.google.com/drive/u/0/folders/1oJJnwBga_XsdvNpYRRZV3BWIOIJnrxO7)

## Paper Highlights

- Introduces ImageNet-BG, a benchmark for background robustness
- Provides a customizable toolkit to generate new dataset variants
- Includes Human Annotations for Quality Assessment (HAQA) annotations for human-recognizable quality filtering
- Benchmarks popular CNN and ViT models
- Shows some classes suffer drastic accuracy drops when backgrounds change

## Citation

To be added upon publication.