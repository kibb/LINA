# Label-free Identification of Neurodegenerative-Disease-Associated Aggregates (LINA)

![Figure 1 - Copy](https://github.com/kibb/LINA/assets/17764656/2e2520dc-2147-45a6-81f9-5e96ec7bb4e0)

This is the repository for [Label-free Identification of Protein Aggregates Using Deep Learning](https://www.biorxiv.org/content/10.1101/2023.04.21.537833v2) by Khalid A. Ibrahim, Kristin S. Grußmayer, Nathan Riguet, Lely Feletti, Hilal A. Lashuel, and Aleksandra Radenovic.

This README provides information on the system requirements and instructions for running the code.

## System Requirements
To run the code, please ensure that your system meets the following requirements:

We recommend a graphics card with > 10 GB of RAM. We used an NVIDIA GeForce RTX 3090.

There are no specific requirements on the operating system. We used Windows Server 2019 Standard.

### Software Dependencies and tested versions: 

We used Python 3.7, Tensorflow 2.8, Keras 2.8, CUDA 11.1 and CUDNN 8.1, but other versions can also be used if compatible. There may be other minor dependencies (packages that are imported in the Python files).

Typical install time is just the time needed to install these dependencies, which if done from scratch should be < 15-30 minutes.

## Contents
In this repository, we provide three pieces of code, with each one provided both as a Jupyter notebook file (.ipynb), where each cell can be run separately, and as a .py file which can be executed all-at-once:
1. LINA_Train: this can be used to train a new U-Net model from scratch using pairs of images (inputs and labels).
2. LINA_Test: this can be used for inference on new transmitted-light (TL) images using one of our pre-trained models.
3. LINA_Transfer: this can be used to apply transfer-learning on the pre-trained model with new pairs of TL images and fluorescent labels.

We provide four pre-trained models requiring different input types, inside the folder 'Models':
1. PixelRegressionModel: a pixel-regression neural network model that expects 8-plane QPI inputs
2. PixelRegressionModel_1Plane: a pixel-regression neural network model that expects 1-plane QPI inputs
3. PixelRegressionModel_BF: a pixel-regression neural network model that expects 8-plane brightfield inputs
4. PixelRegressionModel_BF_1Plane: a pixel-regression neural network model that expects 1-plane brightfield inputs

After installing the necessary system dependencies/environment, running the code is fairly simple. Please make sure the paths for the test data and the models are specified correctly, and to choose the model that is required. By default, PixelRegressionModel is loaded in the code. 

The typical running time on a desktop computer should be around a few minutes. The actual time varies depending on your system configuration.

## Citation

If you use this code, please cite our [pre-print](https://www.biorxiv.org/content/10.1101/2023.04.21.537833v2):

Ibrahim K. A., Grußmayer K. S., Riguet N., Feletti L., Lashuel H. A., Radenovic, A. (2023). Label-free Identification of Protein Aggregates Using Deep Learning. bioRxiv, 2023-04.

```
@article{Ibrahim2023,
  title={Label-free Identification of Protein Aggregates Using Deep Learning},
  author={Ibrahim, Khalid A and Gru{\ss}mayer, Kristin S and Riguet, Nathan and Feletti, Lely and Lashuel, Hilal A and Radenovic, Aleksandra},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

Khalid Ibrahim - khalid.ibrahim@epfl.ch  
Hilal Lashuel - hilal.lashuel@epfl.ch  
Aleksandra Radenovic - aleksandra.radenovic@epfl.ch
