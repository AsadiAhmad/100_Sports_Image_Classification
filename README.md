# 100_Sports_Image_Classification
A deep learning project for sport image classification using a custom VGG19-based architecture with integrated Grad-CAM heatmap visualization for model interpretability.

<div display=flex align=center>
  <img src="/Pictures/Classes.png"/>
</div>

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/TQDM/TQDM.png" title="TQDM" alt="TQDM" width="40" height="40"/>&nbsp;
  <img src="https://www.svgrepo.com/show/373541/cuda.svg" title="CUDA" alt="CUDA" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/pytorch/pytorch.png" title="PyTorach" alt="PyTorach" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/torchvision/torchvision.png" title="TorachVision" alt="TorachVision" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/Kaggle/Kaggle.png" title="Kaggle" alt="Kaggle" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/pillow/pillow.png"  title="pillow" alt="pillow" width="40" height="40"/>&nbsp;
</div>

- Python: Popular language for implementing neural networks and AI projects.
- Jupyter Notebook: Best tool for running Python code cell by cell in an interactive environment.
- Google Colab: Free cloud platform for running Jupyter Notebooks with GPU/TPU support and no local setup needed.
- OpenCV: Powerful library for image processing, computer vision, and real-time applications.
- NumPy: Essential library for numerical operations and array-based computing in Python.
- TQDM: Lightweight library for adding smart progress bars to loops and iterable tasks.
- CUDA: NVIDIA's parallel computing platform for accelerating deep learning on GPUs.
- PyTorch: Deep learning framework known for flexibility, dynamic computation graphs, and strong GPU support.
- TorchVision: Companion library to PyTorch for image-related deep learning tasks, datasets, and transforms.
- Kaggle: Online platform for data science competitions, datasets, and collaborative coding.
- Matplotlib: Versatile library for creating static, animated, and interactive plots in Python.
- Pillow: Python Imaging Library (PIL) fork used for opening, editing, and saving images easily.

## 💻 Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/AsadiAhmad/100_Sports_Image_Classification/blob/main/Code/100_Sports_Image_Classification.ipynb)

## Structure

We have used VGG19 model. also we freeze the 15th first layer.

| **Model Type / Technique** | **Used in Your Code**                           | **Library / Source**                       |
| -------------------------- | ----------------------------------------------- | ------------------------------------------ |
| **CNN (Convolutional NN)** | `VGG19` as the base architecture                | `torchvision.models`                       |
| **Pretrained Model**       | `vgg19(weights=VGG19_Weights.DEFAULT)`          | `torchvision.models`                       |
| **Transfer Learning**      | Using pretrained VGG19 with partial freezing    | `torch.nn`, `torchvision`                  |
| **Freezing Layers**        | First 15 layers of `vgg.features` frozen        | `torch.nn.Parameter.requires_grad = False` |
| **Custom NN (Classifier)** | Custom `nn.Sequential` fully connected layers   | `torch.nn`                                 |
| **Grayscale Input Conv**   | Modified Conv2D to accept grayscale input       | `torch.nn.Conv2d`                          |
| **Dropout**                | `nn.Dropout`, `nn.Dropout2d` for regularization | `torch.nn`                                 |
| **Batch Normalization**    | `nn.BatchNorm1d` for stable training            | `torch.nn`                                 |
| **Activation Function**    | `nn.ReLU` used in classifier                    | `torch.nn`                                 |
| **Weight Initialization**  | Xavier (Glorot) init in classifier              | `torch.nn.init`                            |
| **Adaptive Pooling**       | `nn.AdaptiveAvgPool2d((7,7))`                   | `torch.nn`                                 |
| **Flatten Layer**          | `nn.Flatten()`                                  | `torch.nn`                                 |






















