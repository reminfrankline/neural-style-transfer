# Neural Style Transfer with TensorFlow and VGG19

## Overview
This project implements Neural Style Transfer using TensorFlow and VGG19, a convolutional neural network. The goal is to combine the content of one image with the style of another to create unique, artistic images.

## Project Highlights
- **Objective**: Blend the content of one image with the style of another to produce visually appealing artwork.
- **Technology**: Utilises TensorFlow, VGG19, and Keras for deep learning.
- **Outcome**: Generates new images that capture the style of the reference image while maintaining the content structure of the content image.

## Features
- **Content and Style Extraction**: Uses VGG19 layers to extract features representing content and style.
- **Loss Function**: Computes content loss and style loss to optimise the generated image.
- **Optimisation**: Uses Adam optimiser to iteratively update the generated image.

## Key Learnings
- Understanding the architecture and applications of convolutional neural networks.
- Hands-on experience with deep learning frameworks such as TensorFlow and Keras.
- Practical skills in image processing and computer vision techniques.

## Challenges Overcome
- Managed long processing times by experimenting with different hyper-parameters.
- Ran computations efficiently on available hardware, considering scalability with cloud or supercomputing resources.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- TensorFlow Hub
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

### Installation
Clone the repository and install the required libraries:

```sh
git clone https://github.com/reminfrankline/neural-style-transfer.git
cd neural-style-transfer
pip install -r requirements.txt
