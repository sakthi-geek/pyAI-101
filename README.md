# pyAI-101: A Journey Through the Evolution of AI/ML

Welcome to **pyAI-101**, an educational open-source project aimed at providing a comprehensive understanding of Artificial Intelligence (AI) and Machine Learning (ML) through historical and hands-on perspectives. This project will guide you through the key milestones, mathematical foundations, and seminal algorithms in AI/ML, implemented from scratch in Python. Whether you are a beginner, intermediate, or advanced learner, pyAI-101 is designed to cater to all levels of understanding.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Introduction

### Aim of the Project

**pyAI-101** aims to:

- Provide a historical and chronological journey through the field of AI/ML.
- Implement key algorithms and models from scratch, enabling a deep understanding of their workings.
- Offer a high-level abstraction similar to popular frameworks like scikit-learn and PyTorch, while allowing users to dive into the lower-level details.
- Cater to learners at all levels by providing a modular, well-documented, and user-friendly codebase.

### Why pyAI-101?

- **Historical Perspective**: Understand the evolution of AI/ML through a chronological timeline of key milestones and seminal papers.
- **From Scratch Implementations**: Learn by implementing algorithms and models from the ground up, ensuring a deep understanding of the underlying principles.
- **High-Level Abstraction**: Use an API inspired by popular frameworks, making it easy to switch to or from libraries like scikit-learn and PyTorch.
- **Modular and Extensible**: The codebase is organized into modular components, allowing for easy extension and experimentation.

## Project Structure

```plaintext
pyAI-101/
│
├── README.md
├── license.txt
├── installation_requirements.txt
│
├── 00_historical_timeline/
│   ├── README.md
│   ├── chronological_overview.md
│   └── key_milestones.xlsx
│
├── 01_mathematical_foundations/
│   ├── README.md
│   ├── 01_linear_algebra.py
│   ├── 02_calculus.py
│   ├── 03_statistics.py
│   └── 04_probability.py
│
├── 02_machine_learning/
│   ├── README.md
│   ├── 01_supervised_learning/
│   │   ├── 01_linear_regression.py  
│   │   ├── 02_logistic_regression.py  
│   │   ├── 03_decision_trees.py  
│   │   ├── 04_naive_bayes.py  
│   │   ├── 05_KNN.py  
│   │   └── 06_SVM.py 
│   ├── 02_unsupervised_learning/
│   │   ├── 01_K_means_clustering.py 
│   │   ├── 02_PCA.py
│   │   └── 03_hierarchical_clustering.py
│   └── 03_ensemble_methods/
│       ├── 01_random_forest.py
│       └── 02_gradient_boosting.py
│
├── 03_deep_learning/
│   ├── README.md
│   └── key_architectures/
│       ├── 00_nn_layers.py
│       ├── 01_perceptron.py
│       ├── 02_MLP.py
│       ├── 03_CNN.py
│       ├── 04_ResNet.py
│       ├── 05_EfficientNet.py
│       ├── 06_RNN.py
│       ├── 07_LSTM.py
│       ├── 08_GRU.py
│       ├── 09_attention.py
│       └── 10_autoencoder.py
│
├── 04_reinforcement_learning/
│   ├── README.md
│   ├── 01_traditional_methods/
│   │   ├── 01_Q-learning.py
│   │   ├── 02_SARSA.py
│   │   └── 03_monte_carlo.py
│   ├── 02_deep_rl_methods/
│   │   ├── 01_deep_Q-learning.py
│   │   ├── 02_policy_gradient.py
│   │   ├── 03_actor_critic.py
│   │   └── 04_A3C.py
│   ├── 03_multi_agent_systems/
│   │   ├── 01_cooperative_agents.py
│   │   ├── 02_competitive_agents.py
│   └── exploration_strategies.py
│
├── 05_generative_ai/
│   ├── README.md
│   └── key_architectures/
│       ├── 01_GAN.py
│       ├── 02_VAE.py
│       ├── 03_pixelCNN.py 
│       ├── 04_transformer.py
│       └── 05_diffusion.py
│
├── common_components/
│   ├── README.md
│   ├── 00_backpropagation.py
│   ├── 01_activation_functions.py
│   ├── 02_loss_functions.py
│   ├── 03_optimizers.py
│   ├── 04_evaluation_metrics.py
│   └── 05_utils.py
│
└── resources/
    ├── seminal_papers/
    └── external_links.md
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/pyAI-101.git
    cd pyAI-101
    ```

2. **Install Required Libraries**:
    ```bash
    pip install -r installation_requirements.txt
    ```

## Usage

### Example: Running a Linear Regression Model

Here's how you can use the Linear Regression implementation from scratch:

```python
from pyAI_101.machine_learning.supervised_learning.linear_regression import LinearRegression

# Sample data
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [2, 3, 4, 5]

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict([[5, 6]])
print(predictions)

### Exploring Deep Learning Architectures

You can explore various deep learning architectures like CNN, RNN, and Transformers:

```python
from pyAI_101.deep_learning.key_architectures.cnn import ConvolutionalNeuralNetwork

# Initialize the CNN model
cnn_model = ConvolutionalNeuralNetwork()

# Define your data and train the model
# ...

## Contributing

I welcome contributions from the community! If you would like to contribute, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature-name
    ```
3. **Commit your changes**:
    ```bash
    git commit -m 'Add some feature'
    ```
4. **Push to the branch**:
    ```bash
    git push origin feature-name
    ```
5. **Open a pull request**.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [license.txt](license.txt) file for details.

## Acknowledgements

- Inspired by numerous open-source projects and educational resources.

---

Happy learning! If you have any questions or feedback, feel free to open an issue or contact me at geeksakthi.uk@gmail.com.

