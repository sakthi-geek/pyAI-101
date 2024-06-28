# pyAI-101: A Journey Through the Evolution of AI

Welcome to **pyAI-101**, an educational project designed to guide learners through the historical and technical evolution of Artificial Intelligence (AI). This project will guide you through the key milestones, mathematical foundations, and seminal algorithms in AI/ML. This project provides from-scratch implementations of significant AI algorithms and models, tracing their development from the earliest forms of machine learning to contemporary deep learning and reinforcement learning techniques. Whether you are a beginner, intermediate, or advanced learner, pyAI-101 is designed to cater to all levels of understanding.

![GitHub license](https://img.shields.io/github/license/sakthi-geek/pyAI-101)
![GitHub stars](https://img.shields.io/github/stars/sakthi-geek/pyAI-101?style=social)

## Table of Contents

- [Introduction](#introduction)
- [Project Aim](#project-aim)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## Introduction

## Project Aim

The project started as a personal learning project doing from scratch implementations of AI models/architectures but evolved into more of a learning resource that I want to share with the community. 

**pyAI-101** aims to:

- Provide a historical and chronological journey through the field of AI/ML.
- Implement key algorithms and models from scratch, enabling a deep understanding of their workings.
- Offer a high-level abstraction similar to popular frameworks like scikit-learn and PyTorch, while allowing users to dive into the lower-level details necessary to fully understand the underlying mechanisms.
- Cater to learners at all levels by providing a modular, well-documented, and user-friendly codebase.

### Why pyAI-101?

- **Historical Perspective**: Understand the evolution of AI/ML through a chronological timeline of key milestones and seminal papers.
- **From Scratch Implementations**: Learn by implementing algorithms and models from the ground up, ensuring a deep understanding of the underlying principles.
- **High-Level Abstraction**: Use an API inspired by popular frameworks, making it easy to switch to or from libraries like scikit-learn and PyTorch.
- **Modular and Extensible**: The codebase is organized into modular components, allowing for easy extension and experimentation.

## Project Structure

The project is structured into several directories, each focusing on different aspects of AI:

- `00_AI_evolution_tracker`: Contains documents and spreadsheets that outline the historical development of AI technologies and the key milestones along the way.
- `01_mathematical_foundations`: Modules for basic mathematical tools needed in AI, such as linear algebra and calculus.
- `02_machine_learning`: Implementations of foundational machine learning algorithms like linear regression and SVMs.
- `03_deep_learning`: From classic networks like MLPs and CNNs to modern architectures like Transformers.
- `04_reinforcement_learning`: Covers everything from basic Q-learning to advanced multi-agent systems.
- `05_generative_ai`: Includes implementations of GANs, VAEs, and other generative techniques.
- `pyAI`: Core components, including tensor, autograd, neural network layers, optimization, and utility functions.
- `resources`: Collection of seminal papers and useful external links.

```plaintext
pyAI-101/
│
├── README.md
├── license.txt
├── installation_requirements.txt
│
├── 00_AI_evolution_tracker/
│ ├── README.md
│ ├── chronological_overview.md
│ └── key_milestones.xlsx
│
├── 01_mathematical_foundations/
│ ├── README.md
│ ├── 01_linear_algebra.py
│ ├── 02_calculus.py
│ ├── 03_statistics.py
│ └── 04_probability.py
│
├── 02_machine_learning/
│ ├── README.md
│ ├── 01_supervised_learning/
│ │ ├── 01_linear_regression.py
│ │ ├── 02_logistic_regression.py
│ │ ├── 03_decision_trees.py
│ │ ├── 04_naive_bayes.py
│ │ ├── 05_KNN.py
│ │ └── 06_SVM.py
│ ├── 02_unsupervised_learning/
│ │ ├── 01_K_means_clustering.py
│ │ ├── 02_PCA.py
│ │ └── 03_hierarchical_clustering.py
│ └── 03_ensemble_methods/
│   ├── 01_random_forest.py
│   └── 02_gradient_boosting.py
│
├── 03_deep_learning/
│ ├── README.md
│ └── key_architectures/
│   ├── 01_perceptron.py
│   ├── 02_MLP.py
│   ├── 03_CNN.py
│   ├── 04_ResNet.py
│   ├── 05_EfficientNet.py
│   ├── 06_RNN.py
│   ├── 07_LSTM.py
│   ├── 08_GRU.py
│   ├── 09_autoencoder.py
│   └── 10_transformer.py
│
├── 04_reinforcement_learning/
│ ├── README.md
│ ├── 01_traditional_methods/
│ │ ├── 01_Q-learning.py
│ │ ├── 02_SARSA.py
│ │ └── 03_monte_carlo.py
│ ├── 02_deep_rl_methods/
│ │ ├── 01_deep_Q-learning.py
│ │ ├── 02_policy_gradient.py
│ │ ├── 03_actor_critic.py
│ │ └── 04_A3C.py
│ ├── 03_multi_agent_systems/
│ │ ├── 01_cooperative_agents.py
│ │ └── 02_competitive_agents.py
│ └── exploration_strategies.py
│
├── 05_generative_ai/
│ ├── README.md
│ └── key_architectures/
│   ├── 01_GAN.py
│   ├── 02_VAE.py
│   ├── 03_pixelCNN.py
│   ├── 04_transformer.py
│   └── 05_DDPM.py
│
├── pyAI/
│ ├── README.md
│ ├── __init__.py
│ ├── config.ini
│ ├── main.py
│ ├── autograd/
│ │ ├── autograd.py
│ │ └── tensor.py
│ ├── data/
│ │ ├── dataset.py
│ │ └── data_loader.py
│ ├── nn/
│ │ ├── __init__.py 
│ │ ├── module.py 
│ │ ├── loss.py 
│ │ ├── activation.py 
│ │ └── layers/
│ │   └── linear.py
│ ├── optim/
│ │ ├── __init__.py 
│ │ ├── optimizer.py 
│ │ └── sgd.py 
│ └── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
│
└── resources/
  ├── seminal_papers/
  └── external_links.md
```

### Core Components (`pyAI` directory)

The directory structure and custom implementation purposefully mimics that of the popular deep learning framework PyTorch making it easier for learners to understand how PyTorch works under the hood and helps them to smoothly transition over to the PyTorch framework for further research/experimentation.

- **`autograd`**: Custom automatic differentiation engine built from scratch.
- **`nn`**: Neural network modules, including layers, activations, and loss functions.
- **`optim`**: Optimizers such as SGD and Adam.
- **`data`**: Data loading and preprocessing utilities.
- **`utils`**: Additional utilities like metrics and visualization tools.

```plaintext
├── pyAI/
│ ├── README.md
│ ├── __init__.py
│ ├── config.ini
│ ├── main.py
│ ├── autograd/
│ │ ├── autograd.py
│ │ └── tensor.py
│ ├── data/
│ │ ├── dataset.py
│ │ └── data_loader.py
│ ├── nn/
│ │ ├── __init__.py 
│ │ ├── module.py 
│ │ ├── loss.py 
│ │ ├── activation.py 
│ │ └── layers/
│ │   └── linear.py
│ ├── optim/
│ │ ├── __init__.py 
│ │ ├── optimizer.py 
│ │ └── sgd.py 
│ ├── utils/
│ │ ├── __init__.py
│ │ ├── metrics.py
│ │ └── visualization.py
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Important packages - `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `torchmetrics`, `torchvision`, `torchtext`
- Please see [requirements.txt](requirements.txt) for a list of libraries needed to run the code.

### Installation

Clone this repository and install the required packages

```bash
git clone https://github.com/your-github/pyAI-101.git
cd pyAI-101
pip install -r installation_requirements.txt
```

## Usage

Each module can be run independently to demonstrate the functionality of a specific AI technique. For detailed usage, refer to the `README.md` files within each directory and the documentation provided in the code.

## Contributing

Contributions are welcome! If you have improvements or bug fixes, please fork the repository and submit a pull request.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [license.txt](LICENSE) file for details.

## Authors

- **Sakthignanavel Palanisamy**

## Acknowledgements

- Inspired by numerous open-source projects and educational resources.

---

Happy learning! If you have any questions or feedback, feel free to open an issue or contact me at geeksakthi.uk@gmail.com.