# Autoencoders-FashionMNIST-MNIST
Implementation of autoencoders on Fashion-MNIST and MNIST datasets for image compression and reconstruction, showcasing both shallow and deep architectures.

This repository contains implementations of **autoencoders** for image reconstruction tasks using the **Fashion-MNIST** and **MNIST digit** datasets.  
It demonstrates:
- Data preprocessing
- Building and training simple & deep autoencoders
- Visualizing reconstruction results

 Features
- **Fashion-MNIST Autoencoder**
  - Single hidden layer encoder & decoder
  - Trains in ~10 epochs
  - Visualizes original vs reconstructed images
- **MNIST Autoencoder**
  - Deeper architecture with multiple encoding & decoding layers
  - Uses Mean Squared Error (MSE) loss
  - Achieves significant compression in the latent space
- **Visualization**
  - Side-by-side display of original and reconstructed images
  - Random sample preview

 Datasets
1. **Fashion-MNIST**
   - 60,000 training images, 10,000 test images
   - 28x28 grayscale images of clothing items
2. **MNIST Digits**
   - 60,000 training images, 10,000 test images
   - 28x28 grayscale images of handwritten digits

Both datasets are loaded directly from `tensorflow.keras.datasets`.

 Usage
 Clone the repository
```bash
git clone https://github.com/<your-username>/Autoencoders-FashionMNIST-MNIST.git
cd Autoencoders-FashionMNIST-MNIST

2. Install dependencies

pip install tensorflow matplotlib numpy pandas seaborn

3. Run the notebook

Open AutoEncodersL4.ipynb in Jupyter Notebook or Google Colab and run all cells.
🛠 Model Architectures
Fashion-MNIST Autoencoder

    Encoder: 784 → 128 (ReLU)

    Decoder: 128 → 784 (Sigmoid)

    Loss Function: Binary Crossentropy

    Optimizer: Adam

MNIST Autoencoder

    Encoder: 784 → 100 → 50 → 25 → 2 (ReLU)

    Decoder: 2 → 25 → 50 → 100 → 784 (ReLU)

    Loss Function: Mean Squared Error (MSE)

    Optimizer: Adam

Training Results

    Fashion-MNIST loss converges to ~0.26 after 11 epochs.

    MNIST loss drops from ~4492 to ~2516 after 50 epochs.

 Sample Reconstruction

Example side-by-side comparison of original and reconstructed images:
Original	Reconstructed

 Requirements

    Python 3.8+

    TensorFlow 2.x

    NumPy

    Matplotlib

    Seaborn

    Pandas
