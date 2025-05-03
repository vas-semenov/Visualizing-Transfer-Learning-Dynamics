"""
MLP MNIST Visualization Module
===============================================

This module implements visualization techniques for analyzing the behavior of an MLP trained on the MNIST dataset, based on the methodology described in
"Visualizing the Hidden Activity of Artificial Neural Networks" (Rauber et al., 2017).

Key Components:
    - MLP: The neural network architecture with activation capturing
    - TrainingManager: Handles training and evaluation
    - AdvancedVisualization: Creates visualizations of network behavior
    - LayerwiseVisualization: Analyzes representations at different layers
"""
import torch
import os
import copy
import time
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from scipy.stats import pearsonr
import seaborn as sns
import time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import json

# Random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

"""
MNIST Data Configuration
-----------------------
The MNIST dataset consists of 28x28 grayscale images of handwritten digits from 0 to 9.
The dataset is converted to PyTorch tensor and normalized using MNIST's mean (0.1307) and std (0.3081)

The data is loaded in batches of 16 for efficient training.
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load training dataset
train_dataset = datasets.MNIST(
    root='./data',      # Data storage location
    train=True,     # Using training split
    download=True,      # Download the dataset
    transform=transform     # Apply normalization
)

# Load test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders with optimized settings
train_loader = DataLoader(
    train_dataset,
    batch_size=16,      # For better generalization the batch size is small
    shuffle=True,       # Randomizing training samples
    pin_memory=True     # Connecting the data transfer to GPU
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    pin_memory=True
)


# Neural Network Architecture
class MLP(nn.Module):
    """
    Multi-Layer Perceptron with Activation Capturing
    ----------------------------------------------

    A neural network that performs classification and captures
    internal activations to generate visualizations. The architecture follows
    the paper's (Rauber et al., 2017) specifications with four hidden layers of 1000 neurons each.

    Architecture:
        Input (784) -> Hidden (1000) -> Hidden (1000) -> Hidden (1000) ->
        Hidden (1000) -> Output (10)

    Each hidden layer includes:
        - Linear transformation
        - ReLU activation
        - Dropout (increasing from 0.2 to 0.5)

    The network captures activations after each ReLU layer using hooks,
    enabling analysis of how representations evolve through the network.

    Attributes:
        layer_activations (defaultdict): Stores activations from each ReLU layer
        activation_hooks (list): Stores PyTorch hooks
        num_relu_layers (int): Count of ReLU layers for reference
    """
    def __init__(self, input_size=784):
        """
        Initialize the MLP with specified input size.

        Args:
            input_size (int): Dimension of input features (default: 784 for MNIST)
        """
        super(MLP, self).__init__()

        # Store activation outputs for visualization
        self.layer_activations = defaultdict(list)

        # Define the layers
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second hidden layer
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Third hidden layer
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Fourth hidden layer
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Output layer
            nn.Linear(1000, 10)
        )

        # Register hooks to capture activations
        self.activation_hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Register Forward Hooks for Activation Capture
        ------------------------------------------

        This method sets up PyTorch hooks that capture the output of each ReLU
        layer during forward passes. These activations are essential for
        visualizing how the network represents data at each layer,
        understanding the progression of feature learning and analyzing neuron specialization

        The hooks store detached, CPU-based copies of activations to avoid issues with memory and enable post-processing.
        """
        def get_activation_hook(name):
            """Creates a hook function for a specific layer"""
            def hook(module, input, output):
                self.layer_activations[name].append(output.detach().cpu())

            return hook

        # Register hooks for each ReLU layer
        relu_count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                relu_count += 1
                # Store the layer number with the hook
                hook = module.register_forward_hook(
                    get_activation_hook(f'relu_{relu_count}')
                )
                self.activation_hooks.append(hook)

        # Store the number of ReLU layers for later reference
        self.num_relu_layers = relu_count

    def clear_activations(self):
        """
        Clear stored activations to free memory.
        Should be called after processing each batch.
        """
        self.layer_activations.clear()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784)

        Returns:
            torch.Tensor: Class logits of shape (batch_size, 10)
        """
        x = x.view(x.size(0), -1)
        return self.layers(x)


class TrainingManager:
    """
    Training Management and Monitoring
    ---------------------------------------

    This class handles the complete training cycle of the neural network,
    including activation capturing and visualization generation.
    It implements the training methodology from Rauber (2017),
    while using some modern training practices for improved performance.

    The manager serves three main purposes: it executes and monitors the training process,
    captures and stores network states and activations and generates visualizations of network behavior.

    Key Components:
        - Epoch-level training and evaluation
        - Periodic activation capturing
        - Automated visualization generation
        - Progress tracking and checkpointing
        - Comprehensive metric logging

    Implementation Details:
        The class uses PyTorch's training method while adding new functionality for activation capture and visualization.
        It maintains separate training and evaluation phases to ensure proper batch normalization behavior.
    """
    def __init__(self, model, criterion, optimizer, device, save_dir='./checkpoints'):
        """
        Initialize the training manager with model and training parameters.

        Args:
            model: Neural network model (MLP instance)
            criterion: Loss function (typically CrossEntropyLoss)
            optimizer: Optimization algorithm (e.g., SGD with momentum)
            device: Computing device (CPU/GPU)
            save_dir (str): Directory for saving checkpoints and visualizations
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Training history storage
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.activation_history = []
        self.visualization = AdvancedVisualization(model, device)
        self.layerwise_viz = LayerwiseVisualization(model, device)

    def train_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch with progress tracking.

        This method:
        1. Sets the model to training mode (enabling dropout)
        2. Processes each batch with forward and backward passes
        3. Updates model parameters
        4. Tracks loss and accuracy metrics
        5. Displays progress with tqdm

        Args:
            train_loader: DataLoader for training data
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, accuracy) for this epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move data to device and prepare input
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return running_loss / len(train_loader), correct / total

    def evaluate(self, test_loader):
        """
        Evaluate model performance on the test set.

        This method:
        1. Sets the model to evaluation mode (disabling dropout)
        2. Computes predictions without gradient tracking
        3. Calculates loss and accuracy metrics

        Args:
            test_loader: DataLoader for test data

        Returns:
            tuple: (average_loss, accuracy) on test set
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return test_loss / len(test_loader), correct / total

    def save_activation_snapshot(self, epoch, loader):
        """
        Capture and store network activations for visualization.

        This method captures the network's internal state at a given epoch,
        enabling analysis of how representations evolve during training.

        Args:
            epoch (int): Current epoch number
            loader: DataLoader for activation capture
        """
        activations, labels = get_layer_activations(self.model, loader)
        self.activation_history.append({
            'epoch': epoch,
            'activations': activations,
            'labels': labels
        })

    def train(self, train_loader, test_loader, epochs=100, activation_capture_frequency=10, dataset_name="MNIST"):
        """
        Execute complete training procedure with visualization generation.

        This is the main training loop that:
        1. Trains for the specified number of epochs
        2. Periodically captures activations
        3. Generates visualizations
        4. Saves checkpoints
        5. Tracks and reports metrics

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            epochs (int): Number of training epochs
            activation_capture_frequency (int): Epochs between activation captures
        """
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Evaluation phase
            test_loss, test_acc = self.evaluate(test_loader)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            # Capture activations periodically
            if epoch % activation_capture_frequency == 0:
                self.save_activation_snapshot(epoch, test_loader)

                # Generate visualizations with dataset and epoch information
                print(f"\nGenerating layer-wise visualization for epoch {epoch}...")
                self.layerwise_viz.visualize_layer_comparisons(
                    test_loader,
                    dataset_name=dataset_name,  # Pass dataset name
                    phase="Source Training",
                    epoch=epoch,  # Pass epoch number
                    save_fig=True
                )

                # Compute and display layer statistics
                stats = self.layerwise_viz.compute_layer_statistics(test_loader)
                print(f"\nLayer Statistics for epoch {epoch}:")
                for layer_name, layer_stats in stats.items():
                    print(f"\nLayer: {layer_name}")
                    for stat_name, value in layer_stats.items():
                        print(f"{stat_name}: {value:.4f}")

            # Print epoch summary
            print(f'\nEpoch {epoch} Summary:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

            # Save checkpoint
            self.save_checkpoint(epoch)

        # Final visualization
        print("\nCreating final layer-wise comparison...")
        self.layerwise_viz.visualize_layer_comparisons(
            test_loader,
            dataset_name="MNIST",  # Specify dataset name
            phase="Source Training",  # Specify phase
            epoch=epochs - 1,  # Specify final epoch
            save_fig=True
        )

    def save_checkpoint(self, epoch):
        """
        Save Model Checkpoint and Training State
        --------------------------------------

        This method saves the complete state of training at a given epoch, which enables
        training resumption from any saved point, analysis of model evolution over time and
        comparison of model states across different epochs.

        The checkpoint includes:
        - Model parameters (weights and biases)
        - Optimizer state (momentum buffers, learning rates)
        - Training metrics (losses and accuracies)
        - Current epoch number

        Args:
            epoch (int): The current epoch number, used for checkpoint identification
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')

    def save_training_history(self):
        """
        Save Complete Training History
        ---------------------------

        Archives the complete training metrics history for later analysis.
        This is crucial for analyzing training dynamics, creating learning curves,
        comparing different training runs and identifying potential issues like overfitting

        The history includes:
        - Training and test losses
        - Training and test accuracies
        - All metrics across all epochs
        """
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f)


class AdvancedVisualization:
    """
    Advanced Network Visualization Techniques
    --------------------------------------

    This class implements sophisticated visualization methods as described in
    Rauber et al. (2017) for analyzing neural network behavior.
    It focuses on evolution of representations across training epochs, relationships between neurons and
    network adaptation patterns.

    The visualizations help understand the following things:
    - How the network's internal representations develop
    - Which neurons work together and why
    - How the network organizes information hierarchically

    Attributes:
        model: The neural network being visualized
        device: Computing device (CPU/GPU)
        epoch_activations: Dictionary storing activations by epoch
        neuron_correlations: Matrix of neuron-to-neuron correlations
    """
    def __init__(self, model, device):
        """
        Initialize visualization system with model and device.

        Args:
            model: Neural network model to visualize
            device: Computing device for processing
        """
        self.model = model
        self.device = device
        self.epoch_activations = {}
        self.neuron_correlations = None

    def capture_epoch_activations(self, epoch, data_loader, num_samples=2000):
        """
        Capture Network State at Specific Epoch
        ------------------------------------

        Records the network's internal representations at a given training epoch.
        This data enables analysis of how representations evolve during training.

        The method runs a subset of data through the network, captures activations from specified layers and
        stores both activations and corresponding labels.

        Args:
            epoch (int): Current epoch number
            data_loader: DataLoader providing samples
            num_samples (int): Number of samples to process (default: 2000)
        """
        activations, labels = get_layer_activations(self.model, data_loader, num_samples)
        self.epoch_activations[epoch] = {
            'activations': activations,
            'labels': labels
        }

    def compute_neuron_correlations(self, data_loader, num_samples=2000):
        """
        Analyze Neuron-to-Neuron Relationships
        ------------------------------------

        This method implements a key analysis technique from Rauber et al. (2017) by computing
        correlations between neurons' activation patterns. Understanding these relationships
        reveals how neurons work together to process information.

        The correlation analysis:
        1. Captures activations for a set of input samples
        2. Computes pairwise correlations between all neurons
        3. Handles special cases like constant neurons (zero variance)
        4. Creates a symmetric correlation matrix
        5. Converts correlations to distances for visualization

        The resulting distance matrix allows identification of neuron groups with similar functions,
        understanding of network modularity and visualization of neuron relationships in 2D space

        Args:
            data_loader: DataLoader providing input samples
            num_samples (int): Number of samples to analyze (default: 2000)

        Returns:
            numpy.ndarray: Distance matrix derived from neuron correlations
        """
        activations, _ = get_layer_activations(self.model, data_loader, num_samples)

        # Compute correlation matrix between neurons
        num_neurons = activations.shape[1]
        correlation_matrix = np.zeros((num_neurons, num_neurons))

        # Compute full correlation matrix
        for i in range(num_neurons):
            for j in range(i + 1):  # Only compute lower triangle
                if np.std(activations[:, i]) == 0 or np.std(activations[:, j]) == 0:
                    correlation = 0
                else:
                    correlation, _ = pearsonr(activations[:, i], activations[:, j])
                    correlation = 0 if np.isnan(correlation) else correlation

                # Make matrix symmetric
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation  # Mirror across diagonal

        # Convert correlations to distances
        distances = 1 - np.abs(correlation_matrix)

        self.neuron_correlations = distances
        return distances

    def visualize_epoch_evolution(self, epochs_to_show=None, max_plots_per_figure=12,
                                  title_prefix="", dataset_name=""):
        """
        Visualize Network Learning Progression
        ----------------------------------

        Creates a series of visualizations showing how the network's internal representations
        evolve throughout training. This implementation extends the methodology from
        Rauber et al. (2017) by adding support for batch processing of epochs and
        automated layout management.

        The visualization process:
        1. Projects high-dimensional activations to 2D using t-SNE
        2. Colors points by their class labels
        3. Creates multiple subplots showing different epochs
        4. Handles large numbers of epochs through batch processing

        The resulting visualizations reveal how class separation develops over time,
        when and how clusters form, the stability of learned representations and
        potential issues like mode collapse or instability

        Args:
            epochs_to_show (list, optional): Specific epochs to visualize
            max_plots_per_figure (int): Maximum number of subplots per figure
            title_prefix (str): Optional prefix for the title (e.g., "Source:" or "Target:")
            dataset_name (str): Name of the dataset (e.g., "MNIST" or "SVHN")
        """
        if epochs_to_show is None:
            epochs_to_show = sorted(self.epoch_activations.keys())

        dataset_info = f" on {dataset_name}" if dataset_name else ""

        # Split epochs into batches
        for i in range(0, len(epochs_to_show), max_plots_per_figure):
            batch_epochs = epochs_to_show[i:i + max_plots_per_figure]

            # Create a figure for this batch
            rows = int(np.ceil(len(batch_epochs) / 2))
            fig = plt.figure(figsize=(15, rows * 5))

            # Create t-SNE projections for each epoch in this batch
            for idx, epoch in enumerate(batch_epochs, 1):
                data = self.epoch_activations[epoch]

                # Compute t-SNE projection
                tsne = TSNE(n_components=2, random_state=42)
                projections = tsne.fit_transform(data['activations'])

                # Create subplot
                ax = fig.add_subplot(rows, 2, idx)
                scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                     c=data['labels'], cmap='tab10', alpha=0.6)
                ax.set_title(f'{title_prefix} Epoch {epoch}')

                if idx == 1:  # Add colorbar only for first plot
                    plt.colorbar(scatter)

            plt.suptitle(
                f'{title_prefix} Evolution of Representations{dataset_info} (Epochs {batch_epochs[0]}-{batch_epochs[-1]})',
                fontsize=16)
            plt.tight_layout()

            # Save figure if we have a dataset name
            if dataset_name:
                phase = title_prefix.strip(':') if title_prefix else "training"
                plt.savefig(f'{dataset_name}_{phase}_epochs_{batch_epochs[0]}-{batch_epochs[-1]}.png')

            plt.show()

    def visualize_neuron_relationships(self, dataset_name="", phase=""):
        """
        Create a Map of Neuron Relationships
        ----------------------------------

        This method implements one of the key visualization techniques from Rauber et al. (2017),
        creating a two-dimensional representation of how neurons relate to each other. The visualization
        reveals the network's internal organization and functional grouping of neurons.

        The visualization process involves:
        1. Using previously computed correlation-based distances between neurons
        2. Applying Multidimensional Scaling (MDS) to create a 2D layout
        3. Coloring neurons based on their average correlation strength
        4. Creating an interactive scatter plot with a correlation strength colorbar

        The resulting visualization reveals groups of neurons that work together,
        hierarchical organization of the network, potential specialization of different network regions and
        overall modularity of the network.

        Args:
            dataset_name (str): Name of the dataset for saving/titling
            phase (str): "Source", "Target", or other descriptor of training phase

        Raises:
            ValueError: If neuron correlations haven't been computed yet
        """
        if self.neuron_correlations is None:
            raise ValueError("Must compute neuron correlations first!")

            # Use MDS to create 2D embedding of neurons
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        neuron_positions = mds.fit_transform(self.neuron_correlations)

        # Create visualization
        plt.figure(figsize=(12, 12))

        # Color neurons based on their average correlation with others
        avg_correlations = np.mean(np.abs(self.neuron_correlations), axis=1)

        scatter = plt.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
                              c=avg_correlations, cmap='viridis',
                              s=100, alpha=0.6)
        plt.colorbar(scatter, label='Average Correlation Strength')

        # Create more descriptive title
        title_parts = []
        if phase:
            title_parts.append(phase)
        title_parts.append("Neuron Relationship Map")
        if dataset_name:
            title_parts.append(f"for {dataset_name}")

        plt.title(" ".join(title_parts))
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')

        # Save figure if we have a dataset name
        if dataset_name:
            phase_str = f"{phase}_" if phase else ""
            plt.savefig(f'{dataset_name}_{phase_str}neuron_map.png')

        plt.show()


class LayerwiseVisualization:
    """
    Layer-by-Layer Network Analysis
    ----------------------------

    This class implements techniques for analyzing and visualizing how information
    is processed at different depths within the network. It extends the methodology
    from Rauber et al. (2017) by providing detailed statistical analysis of layer
    behavior alongside visualizations.

    Key components:
    - Capture and analyze activations from each network layer
    - Compare representation quality across layers
    - Measure class separation at different network depths
    - Track how features become more abstract in deeper layers

    The analysis helps understand:
    - How different layers contribute to the overall network function
    - Where in the network important features emerge
    - How representations become more abstract/specialized
    - Whether all layers are necessary/effective

    Attributes:
        model: The neural network being analyzed
        device: Computing device (CPU/GPU)
        layer_activations: Dictionary storing activations by layer
    """
    def __init__(self, model, device):
        """
        Initialize Layer Analysis System
        ----------------------------

        Sets up the visualization system and registers hooks for capturing
        layer-wise activations.

        Args:
            model: Neural network model to analyze
            device: Computing device for processing
        """
        self.model = model
        self.device = device
        # Dictionary to store activations from all layers
        self.layer_activations = {}
        # Register hooks for all ReLU layers
        self._register_hooks()

    def _register_hooks(self):
        """
        Register Activation Capturing Hooks
        --------------------------------

        This method sets up PyTorch forward hooks to capture activations from each ReLU layer
        in the network. Unlike the activation capturing in the MLP class which stores historical
        activations, these hooks focus on the current state of each layer.

        The hooks intercept the output of each ReLU layer during forward passes,
        store a copy of the activations in CPU memory and organize activations by layer name for later analysis.

        This layerwise capturing is essential for comparing how different network depths
        process the same input data, providing insights into the hierarchical feature
        extraction process described in Rauber et al. (2017).
        """

        def get_activation_hook(name):
            def hook(module, input, output):
                # Store activations by layer name
                self.layer_activations[name] = output.detach().cpu()

            return hook

        # Register hooks for each ReLU layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(get_activation_hook(name))

    def get_layer_representations(self, data_loader, num_samples=2000):
        """
        Collect Comprehensive Layer Activation Data
        ---------------------------------------

        This method gathers and organizes activations from all network layers for a given
        set of input samples. It provides the foundation for all layer-wise analysis
        and visualization by:
        - Running input data through the network in evaluation mode
        - Collecting activations from each layer via the registered hooks
        - Organizing activations by layer with corresponding class labels
        - Managing memory by processing data in batches

        The comprehensive layer data allows
        direct comparison of how the same inputs are represented at different depths,
        analysis of feature abstraction across the network, measurement of class separation at each layer and
        visualization of the network's hierarchical processing.

        This implementation extends the paper's approach by supporting arbitrary
        sampling sizes and batch processing for memory efficiency.

        Args:
            data_loader: DataLoader providing input samples
            num_samples (int): Number of samples to analyze (default: 2000)

        Returns:
            tuple: (all_layers_data, labels) where:
                - all_layers_data is a dictionary mapping layer names to activation arrays
                - labels is a numpy array of corresponding class labels
        """
        self.model.eval()
        all_layers_data = {}
        labels = []
        count = 0

        with torch.no_grad():
            for data, target in data_loader:
                if count >= num_samples:
                    break

                remaining = num_samples - count
                batch_size = min(data.size(0), remaining)

                # Forward pass to collect activations
                data = data.to(self.device)
                _ = self.model(data)

                # Store activations from each layer
                for layer_name, activations in self.layer_activations.items():
                    if layer_name not in all_layers_data:
                        all_layers_data[layer_name] = []
                    # Store only the needed samples
                    layer_data = activations[:batch_size].numpy()
                    all_layers_data[layer_name].append(layer_data)

                labels.extend(target[:batch_size].numpy())
                count += batch_size

        # Concatenate all batches for each layer
        for layer_name in all_layers_data:
            all_layers_data[layer_name] = np.vstack(all_layers_data[layer_name])

        return all_layers_data, np.array(labels)

    def visualize_layer_comparisons(self, data_loader, num_samples=2000,
                                    dataset_name="", phase="", epoch=None, save_fig=True):
        """
        Create Multi-Layer Representation Comparison
        ----------------------------------------

        This method implements a key visualization from Rauber et al. (2017) by creating
        side-by-side comparisons of how different network layers represent the same input data.
        The visualization reveals the progressive transformation of data through the network.

        The visualization process collects activations from all layers for the same set of inputs,
        creates t-SNE projections of each layer's high-dimensional activations,
        arranges projections in a grid layout with consistent coloring and
        adds appropriate titles and labels for interpretation.

        The resulting visualization shows:
        - How class separation improves in deeper layers
        - The emergence of clusters and decision boundaries
        - Layer-by-layer refinement of representations
        - Potential issues like information bottlenecks

        Args:
            data_loader: DataLoader providing input samples
            num_samples (int): Number of samples to visualize (default: 2000)
            dataset_name (str): Name of the dataset for saving/titling
            phase (str): "Source", "Target", or other descriptor of training phase
            epoch (int): Current epoch number
            save_fig (bool): Whether to save the figure to a file
        """
        # Get activations from all layers
        layer_data, labels = self.get_layer_representations(data_loader, num_samples)

        # Create subplot grid
        num_layers = len(layer_data)
        fig_size = min(20, num_layers * 5)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Create t-SNE projection for each layer
        for idx, (layer_name, activations) in enumerate(layer_data.items(), 1):
            # Get friendly layer name
            friendly_name = get_friendly_layer_name(layer_name)

            # Compute t-SNE projection
            tsne = TSNE(n_components=2, random_state=42)
            projections = tsne.fit_transform(activations)

            # Create subplot
            ax = fig.add_subplot(int(np.ceil(num_layers / 2)), 2, idx)
            scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                 c=labels, cmap='tab10', alpha=0.6)
            ax.set_title(f'{friendly_name}')

            if idx == 1:  # Add colorbar only for first plot
                plt.colorbar(scatter)

        # Create detailed suptitle with dataset and epoch information
        title_parts = []

        # Always include dataset name if provided
        if dataset_name:
            title_parts.append(f"Dataset: {dataset_name}")

        # Always include phase if provided
        if phase:
            title_parts.append(f"Phase: {phase}")

        # Always include epoch if provided
        if epoch is not None:
            title_parts.append(f"Epoch: {epoch}")

        # Add base title
        title_parts.append("Layer-wise Representation Comparison")

        # Join all parts with separators
        plt.suptitle(" | ".join(title_parts), fontsize=16)
        plt.tight_layout()

        # Save figure if requested
        if save_fig and dataset_name:
            # Create descriptive filename
            components = []
            components.append(dataset_name.lower())

            if phase:
                components.append(phase.lower().replace(" ", "_"))

            if epoch is not None:
                components.append(f"epoch_{epoch}")

            components.append("layer_comparison")
            filename = "_".join(components) + ".png"

            plt.savefig(filename)

        plt.show()

    def compute_layer_statistics(self, data_loader, num_samples=2000):
        """
        Calculate Quantitative Layer Metrics
        ---------------------------------

        This method extends the visual analysis with statistical measurements that
        quantify each layer's behavior. These metrics provide objective comparisons
        across layers and can reveal patterns that might not be visually obvious.

        The method calculates:
        - Basic activation statistics (mean, std, min, max)
        - Sparsity (proportion of zero activations)
        - Class separation (average distance between class centroids)

        These metrics show which layers contribute most to class separation,
        how does sparsity change through the network, where do the most significant transformations occur and
        whether all layers are effectively utilized?

        Args:
            data_loader: DataLoader providing input samples
            num_samples (int): Number of samples to analyze (default: 2000)

        Returns:
            dict: Statistics for each layer, organized by layer name
        """
        layer_data, labels = self.get_layer_representations(data_loader, num_samples)
        stats = {}

        for layer_name, activations in layer_data.items():
            # Compute various statistical measures
            stats[layer_name] = {
                'mean_activation': np.mean(activations),
                'std_activation': np.std(activations),
                'sparsity': np.mean(activations == 0),  # ReLU sparsity
                'max_activation': np.max(activations),
                'min_activation': np.min(activations)
            }

            # Compute class separation metric
            class_means = []
            for class_idx in np.unique(labels):
                class_data = activations[labels == class_idx]
                class_means.append(np.mean(class_data, axis=0))

            # Average distance between class centers
            class_means = np.array(class_means)
            distances = []
            for i in range(len(class_means)):
                for j in range(i + 1, len(class_means)):
                    dist = np.linalg.norm(class_means[i] - class_means[j])
                    distances.append(dist)

            stats[layer_name]['class_separation'] = np.mean(distances)

        return stats


# Visualization function for training history
def plot_training_progress(trainer, dataset_name="", phase=""):
    """
    Visualize Training Dynamics
    ------------------------

    Creates a dual-plot visualization showing how loss and accuracy evolve during
    training. This visualization helps identify convergence patterns,
    potential overfitting (divergence between training and test curves), learning plateaus and training stability.

    The left plot shows loss curves for both training and test sets, while
    the right plot shows accuracy curves. This side-by-side arrangement allows
    direct comparison of these complementary metrics.

    Args:
        trainer: TrainingManager instance containing training history
        dataset_name (str): Name of the dataset for saving/titling
        phase (str): "Source", "Target", or other descriptor of training phase
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.plot(trainer.test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Create descriptive title
    title_parts = []
    if phase:
        title_parts.append(phase)
    title_parts.append("Training and Test Losses")
    if dataset_name:
        title_parts.append(f"({dataset_name})")
    ax1.set_title(" ".join(title_parts))
    ax1.legend()

    # Plot accuracies
    ax2.plot(trainer.train_accuracies, label='Train Accuracy')
    ax2.plot(trainer.test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    # Create descriptive title
    title_parts = []
    if phase:
        title_parts.append(phase)
    title_parts.append("Training and Test Accuracies")
    if dataset_name:
        title_parts.append(f"({dataset_name})")
    ax2.set_title(" ".join(title_parts))
    ax2.legend()

    plt.tight_layout()

    # Save figure if we have a dataset name
    if dataset_name:
        phase_str = f"{phase}_" if phase else ""
        plt.savefig(f'{dataset_name}_{phase_str}training_progress.png')

    plt.show()

def visualize_layer_adaptation(layer_distances, source_name, target_name, save_fig=True):
    """
    Visualize Layer Adaptation During Transfer Learning
    -----------------------------------------------

    Shows how much each layer changed during transfer from source to target domain.
    Higher bars indicate more adaptation of the layer weights.

    Args:
        layer_distances: Dictionary mapping layer names to their adaptation distances
        source_name: Name of source dataset
        target_name: Name of target dataset
        save_fig: Whether to save the figure to a file
    """
    # Extract layer names and distances
    layers = []
    distances = []

    for layer_name, distance in layer_distances.items():
        # Convert technical layer names to friendly names
        friendly_name = get_friendly_layer_name(layer_name)
        layers.append(friendly_name)
        distances.append(distance)

    # Create sorted pairs for visualization
    pairs = sorted(zip(layers, distances), key=lambda x: x[1])
    layers = [p[0] for p in pairs]
    distances = [p[1] for p in pairs]

    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(layers)), distances, align='center')
    plt.yticks(range(len(layers)), layers)
    plt.xlabel('Adaptation Distance')
    plt.title(f'Layer Adaptation: {source_name} → {target_name}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save figure if requested
    if save_fig:
        plt.savefig(f'adaptation_{source_name}_to_{target_name}.png')

    plt.tight_layout()
    plt.show()


class TransferLearningManager:
    """
    Transfer Learning Management System
    ----------------------------------

    This class handles the complete transfer learning pipeline, including:
    1. Training/loading a source model
    2. Capturing source model representations
    3. Fine-tuning on a target dataset
    4. Tracking representation changes during fine-tuning
    5. Visualizing the transfer learning dynamics

    It builds upon the base TrainingManager but adds specific
    functionality for transfer learning visualization.
    """

    def __init__(self, model, criterion, optimizer, device,
                 source_dataset_name, target_dataset_name,
                 base_save_dir='./transfer_learning'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.source_dataset_name = source_dataset_name
        self.target_dataset_name = target_dataset_name

        # Create directories for saving models and visualizations
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(exist_ok=True)

        # Create source and target directories
        self.source_dir = self.base_save_dir / source_dataset_name
        self.source_dir.mkdir(exist_ok=True)

        self.target_dir = self.base_save_dir / f"{source_dataset_name}_to_{target_dataset_name}"
        self.target_dir.mkdir(exist_ok=True)

        # Storage for tracking transfer learning dynamics
        self.source_activations = {}
        self.target_activations = {}
        self.adaptation_metrics = {}

        # Create visualization tools
        self.layer_viz = LayerwiseVisualization(model, device)

    def train_source_model(self, train_loader, test_loader, epochs=100,
                           activation_capture_frequency=10):
        """
        Train the initial source model and capture its representations

        Args:
            train_loader: DataLoader for source training data
            test_loader: DataLoader for source test data
            epochs: Number of training epochs
            activation_capture_frequency: How often to capture activations
        """
        print(f"Training source model on {self.source_dataset_name}...")

        # Create a trainer for the source model
        trainer = TrainingManager(
            self.model,
            self.criterion,
            self.optimizer,
            self.device,
            save_dir=self.source_dir
        )

        # Train the source model
        trainer.train(train_loader, test_loader, epochs, activation_capture_frequency)

        # Save the final model
        self.save_source_model()

        # Capture the source model's final layer representations
        self._capture_source_representations(test_loader)

        return trainer

    def save_source_model(self):
        """Save the source model's weights and configuration"""
        model_path = self.source_dir / 'source_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dataset': self.source_dataset_name,
            'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S")
        }, model_path)
        print(f"Source model saved to {model_path}")

    def load_source_model(self, model_path=None):
        """
        Load a previously trained source model

        Args:
            model_path: Path to saved model (if None, uses default location)
        """
        if model_path is None:
            model_path = self.source_dir / 'source_model.pt'

        if not model_path.exists():
            raise FileNotFoundError(f"Source model not found at {model_path}")

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded source model trained on {checkpoint['dataset']}")

        return checkpoint

    def _capture_source_representations(self, data_loader, num_samples=2000):
        """Capture and store the source model's layer activations"""
        print("Capturing source model representations...")
        layer_data, labels = self.layer_viz.get_layer_representations(data_loader, num_samples)

        # Store activations for later comparison
        self.source_activations = {
            'layer_data': layer_data,
            'labels': labels
        }

    def fine_tune(self, train_loader, test_loader, epochs=50,
                  layers_to_freeze=None, activation_capture_frequency=5,
                  learning_rate_reduction=0.1, target_input_size=3072):
        """
        Fine-tune the model on the target dataset with careful monitoring
        of how representations change during transfer

        Args:
            train_loader: DataLoader for target training data
            test_loader: DataLoader for target test data
            epochs: Number of fine-tuning epochs
            layers_to_freeze: List of layer indices to freeze (None = fine-tune all)
            activation_capture_frequency: How often to capture activations
            learning_rate_reduction: Factor to reduce learning rate (compared to source)
            target_input_size: Input size for the target dataset (default: 3072 for SVHN)
        """
        print(f"Fine-tuning model for {self.target_dataset_name}...")

        # Adapt model for the target dataset's input size
        self.prepare_model_for_target_dataset(target_input_size)

        # Save initial model state for comparison
        initial_state = copy.deepcopy(self.model.state_dict())

        # Freeze specified layers if requested
        if layers_to_freeze:
            self._freeze_layers(layers_to_freeze)

        # Adjust learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= learning_rate_reduction
            print(f"Fine-tuning learning rate: {param_group['lr']}")

        # Storage for tracking adaptation
        self.target_activations = {}
        self.adaptation_metrics = {
            'layer_distances': [],
            'epoch_adaptation_speed': []
        }

        # Fine-tuning loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # Evaluation phase
            test_loss, test_acc = self._evaluate(test_loader)

            # Capture activations periodically
            if epoch % activation_capture_frequency == 0 or epoch == epochs - 1:
                # Capture current representations
                layer_data, labels = self.layer_viz.get_layer_representations(test_loader)

                # Store for later analysis
                self.target_activations[epoch] = {
                    'layer_data': layer_data,
                    'labels': labels
                }

                # Compute adaptation metrics
                self._compute_adaptation_metrics(epoch, initial_state)

                # Visualize the current state
                self._visualize_transfer_state(epoch)

            # Print epoch summary
            print(f'\nFine-tuning Epoch {epoch} Summary:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

            # Save checkpoint
            self._save_fine_tuning_checkpoint(epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            })

        # Final visualization and analysis
        self._create_final_visualizations()

    def _freeze_layers(self, layers_to_freeze):
        """Freeze specified layers to prevent updating during fine-tuning"""
        if not isinstance(layers_to_freeze, list):
            layers_to_freeze = [layers_to_freeze]

        # Get all named parameters
        for name, param in self.model.named_parameters():
            # Freeze parameters in specified layers
            for layer_idx in layers_to_freeze:
                layer_name = f"layers.{layer_idx * 3}"  # Adjust for the 3 components per layer
                if layer_name in name:
                    param.requires_grad = False
                    print(f"Freezing {name}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    def _train_epoch(self, train_loader, epoch):
        """Train the model for one fine-tuning epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return running_loss / len(train_loader), correct / total

    def _evaluate(self, test_loader):
        """Evaluate the fine-tuned model on test data"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return test_loss / len(test_loader), correct / total

    def _compute_adaptation_metrics(self, epoch, initial_state):
        """
        Compute metrics to quantify how much different layers have adapted
        during transfer learning
        """
        # Calculate weight changes from initial state
        layer_distances = {}

        for name, current_params in self.model.named_parameters():
            if name in initial_state:
                # Calculate Euclidean distance between initial and current parameters
                initial_params = initial_state[name]
                distance = torch.norm(current_params - initial_params).item()
                layer_distances[name] = distance

        # Store metrics
        self.adaptation_metrics['layer_distances'].append({
            'epoch': epoch,
            'distances': layer_distances
        })

        # Calculate representation dissimilarity between source and current
        if self.source_activations and epoch in self.target_activations:
            representation_changes = {}

            for layer_name in self.source_activations['layer_data'].keys():
                if layer_name in self.target_activations[epoch]['layer_data']:
                    source_acts = self.source_activations['layer_data'][layer_name]
                    target_acts = self.target_activations[epoch]['layer_data'][layer_name]

                    # For simplicity, calculate mean Euclidean distance between centroids
                    source_centroid = np.mean(source_acts, axis=0)
                    target_centroid = np.mean(target_acts, axis=0)

                    # Calculate centroid distance
                    centroid_distance = np.linalg.norm(source_centroid - target_centroid)
                    representation_changes[layer_name] = centroid_distance

            self.adaptation_metrics['epoch_adaptation_speed'].append({
                'epoch': epoch,
                'representation_changes': representation_changes
            })

    def _save_fine_tuning_checkpoint(self, epoch, metrics):
        """Save checkpoint during fine-tuning"""
        checkpoint_path = self.target_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)

    def _visualize_transfer_state(self, epoch):
        """
        Create visualizations to show the current state of transfer learning
        for a given epoch
        """
        if epoch not in self.target_activations:
            return

        # Generate layer-wise visualizations for current state
        print(f"\nVisualizing transfer learning state at epoch {epoch}...")

        # Get source and target activations
        target_data = self.target_activations[epoch]['layer_data']
        target_labels = self.target_activations[epoch]['labels']

        # Plot layer-wise representations for target data with dataset name and epoch
        self._visualize_layer_representations(
            target_data,
            target_labels,
            epoch
        )

        # If we have adaptation metrics, visualize them
        if self.adaptation_metrics['layer_distances']:
            self._visualize_adaptation_metrics(epoch)

    def _visualize_layer_representations(self, layer_data, labels, epoch):
        """Create visualizations of layer representations with dataset and epoch information"""
        num_layers = len(layer_data)
        fig_size = min(20, num_layers * 5)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Create t-SNE projection for each layer
        for idx, (layer_name, activations) in enumerate(layer_data.items(), 1):
            # Get friendly layer name
            friendly_name = get_friendly_layer_name(layer_name)

            # Compute t-SNE projection
            tsne = TSNE(n_components=2, random_state=42)
            projections = tsne.fit_transform(activations)

            # Create subplot
            ax = fig.add_subplot(int(np.ceil(num_layers / 2)), 2, idx)
            scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                 c=labels, cmap='tab10', alpha=0.6)
            ax.set_title(f'{friendly_name}')

            if idx == 1:  # Add colorbar only for first plot
                plt.colorbar(scatter)

        # Include dataset name and epoch information in the title
        plt.suptitle(
            f'Dataset: {self.target_dataset_name} | Phase: Transfer Learning | Epoch: {epoch} | Layer-wise Representation Comparison',
            fontsize=16)
        plt.tight_layout()

        # Save the figure
        filename = f'{self.target_dataset_name}_transfer_learning_epoch_{epoch}_layer_representations.png'
        plt.savefig(self.target_dir / filename)
        plt.close()

    def _visualize_adaptation_metrics(self, epoch):
        """Visualize how much each layer has adapted during transfer"""
        # Find the entry for the current epoch
        epoch_distances = None
        for entry in self.adaptation_metrics['layer_distances']:
            if entry['epoch'] == epoch:
                epoch_distances = entry['distances']
                break

        if not epoch_distances:
            return

        # Create visualization of layer distances
        plt.figure(figsize=(12, 6))

        # Group parameters by layer
        layer_groups = {}
        for name, distance in epoch_distances.items():
            # Extract layer number from parameter name
            if 'layers' in name:
                # Extract layer number assuming format 'layers.X.weight'
                parts = name.split('.')
                if len(parts) >= 2:
                    layer_num = int(parts[1]) // 3  # Convert to logical layer number
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = []
                    layer_groups[layer_num].append(distance)

        # Calculate average distance per layer
        layers = []
        avg_distances = []
        for layer, distances in sorted(layer_groups.items()):
            layers.append(f"Layer {layer}")
            avg_distances.append(np.mean(distances))

        # Create bar chart
        plt.bar(layers, avg_distances)
        plt.xlabel('Network Layer')
        plt.ylabel('Average Parameter Distance from Source')
        plt.title(f'Layer Adaptation After {epoch} Fine-tuning Epochs')
        plt.xticks(rotation=45)

        # Save the figure
        plt.tight_layout()
        plt.savefig(self.target_dir / f'layer_adaptation_epoch_{epoch}.png')
        plt.close()

    def _create_final_visualizations(self):
        """Create comprehensive visualizations of the entire transfer process"""
        # Create adaptation trajectory visualization
        self._visualize_adaptation_trajectory()

        # Create comparison of source vs. final target representations
        self.visualize_transfer_comparison()

        # Create layer adaptation visualization
        self.visualize_layer_adaptation()

        # Create feature preservation/adaptation visualization
        self._visualize_feature_transfer()

    def _visualize_adaptation_trajectory(self):
        """Visualize how representations evolved throughout fine-tuning"""
        if not self.adaptation_metrics['epoch_adaptation_speed']:
            return

        # Extract data
        epochs = []
        layer_changes = {}

        for entry in self.adaptation_metrics['epoch_adaptation_speed']:
            epochs.append(entry['epoch'])
            for layer, change in entry['representation_changes'].items():
                if layer not in layer_changes:
                    layer_changes[layer] = []
                layer_changes[layer].append(change)

        # Create line chart
        plt.figure(figsize=(12, 8))

        for layer, changes in layer_changes.items():
            # Make sure changes list matches epochs length
            if len(changes) == len(epochs):
                plt.plot(epochs, changes, marker='o', label=f'Layer {layer}')

        plt.xlabel('Fine-tuning Epoch')
        plt.ylabel('Representation Change (Distance from Source)')
        plt.title('Layer Representation Evolution During Transfer Learning')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(self.target_dir / 'adaptation_trajectory.png')
        plt.close()

    def _visualize_source_target_comparison(self):
        """Create side-by-side comparison of source and final target representations"""
        if not self.source_activations or not self.target_activations:
            return

        # Get final target epoch
        final_epoch = max(self.target_activations.keys())

        source_layer_data = self.source_activations['layer_data']
        source_labels = self.source_activations['labels']

        target_layer_data = self.target_activations[final_epoch]['layer_data']
        target_labels = self.target_activations[final_epoch]['labels']

        # Create visualizations for each layer
        for layer_name in source_layer_data.keys():
            if layer_name in target_layer_data:
                plt.figure(figsize=(16, 8))

                # Source representation
                plt.subplot(1, 2, 1)
                source_acts = source_layer_data[layer_name]
                tsne = TSNE(n_components=2, random_state=42)
                source_proj = tsne.fit_transform(source_acts)

                plt.scatter(source_proj[:, 0], source_proj[:, 1],
                            c=source_labels, cmap='tab10', alpha=0.6)
                plt.title(f'Source: {self.source_dataset_name} ({layer_name})')

                # Target representation
                plt.subplot(1, 2, 2)
                target_acts = target_layer_data[layer_name]
                tsne = TSNE(n_components=2, random_state=42)
                target_proj = tsne.fit_transform(target_acts)

                plt.scatter(target_proj[:, 0], target_proj[:, 1],
                            c=target_labels, cmap='tab10', alpha=0.6)
                plt.title(f'Target: {self.target_dataset_name} ({layer_name})')

                plt.suptitle(f'Source vs. Target Representations: {layer_name}')
                plt.tight_layout()

                # Save figure
                plt.savefig(self.target_dir / f'source_target_comparison_{layer_name}.png')
                plt.close()

    def _visualize_feature_transfer(self):
        """
        Create visualization showing which features were preserved
        versus adapted during transfer learning
        """
        # This requires more advanced analysis of the feature space
        # Here we'll implement a simple version that shows overall adaptation
        if not self.adaptation_metrics['layer_distances']:
            return

        # Extract final adaptation state
        final_entry = self.adaptation_metrics['layer_distances'][-1]

        # Get parameter distances grouped by layer
        layer_param_distances = {}
        for name, distance in final_entry['distances'].items():
            layer_name = name.split('.')[1] if len(name.split('.')) > 1 else name
            if layer_name not in layer_param_distances:
                layer_param_distances[layer_name] = []
            layer_param_distances[layer_name].append(distance)

        # Calculate statistics for each layer
        layers = []
        means = []
        stds = []

        for layer, distances in sorted(layer_param_distances.items(),
                                       key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
            layers.append(layer)
            means.append(np.mean(distances))
            stds.append(np.std(distances))

        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(layers, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Layer Parameter')
        plt.ylabel('Parameter Change Distance')
        plt.title('Feature Transfer Analysis: Parameter Changes by Layer')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add horizontal line for average change
        avg_change = np.mean(means)
        plt.axhline(y=avg_change, color='r', linestyle='-',
                    label=f'Average Change: {avg_change:.4f}')
        plt.legend()

        # Save figure
        plt.tight_layout()
        plt.savefig(self.target_dir / 'feature_transfer_analysis.png')
        plt.close()

    def _visualize_source_target_layer_comparison(self, source_activations, source_labels,
                                                  target_activations, target_labels,
                                                  layer_name, target_epoch):
        """
        Create side-by-side visualization comparing source and target representations
        for a specific layer
        """
        # Get friendly layer name
        friendly_name = get_friendly_layer_name(layer_name)

        # Create figure
        plt.figure(figsize=(16, 8))

        # Source representation
        plt.subplot(1, 2, 1)
        tsne = TSNE(n_components=2, random_state=42)
        source_proj = tsne.fit_transform(source_activations)

        plt.scatter(source_proj[:, 0], source_proj[:, 1],
                    c=source_labels, cmap='tab10', alpha=0.6)
        plt.title(f'Source: {self.source_dataset_name} ({friendly_name})')
        plt.colorbar()

        # Target representation
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, random_state=42)
        target_proj = tsne.fit_transform(target_activations)

        plt.scatter(target_proj[:, 0], target_proj[:, 1],
                    c=target_labels, cmap='tab10', alpha=0.6)
        plt.title(f'Target: {self.target_dataset_name} ({friendly_name}) - Epoch {target_epoch}')
        plt.colorbar()

        plt.suptitle(f'Transfer Learning: {self.source_dataset_name} → {self.target_dataset_name} ({friendly_name})',
                     fontsize=16)
        plt.tight_layout()

        # Save figure
        filename = f'transfer_{self.source_dataset_name}_to_{self.target_dataset_name}_{layer_name}_epoch_{target_epoch}.png'
        plt.savefig(self.target_dir / filename)
        plt.close()

    def visualize_transfer_comparison(self):
        """
        Create visualizations comparing source and target representations
        for each layer after transfer learning
        """
        print("\nCreating source vs target comparison visualizations...")

        # Get final target epoch
        final_epochs = list(self.target_activations.keys())
        if not final_epochs:
            print("No target activations available for comparison")
            return

        final_epoch = max(final_epochs)

        # Extract source and target data
        if not self.source_activations:
            print("No source activations available for comparison")
            return

        source_layer_data = self.source_activations['layer_data']
        source_labels = self.source_activations['labels']

        target_layer_data = self.target_activations[final_epoch]['layer_data']
        target_labels = self.target_activations[final_epoch]['labels']

        # For each layer present in both source and target
        for layer_name in source_layer_data.keys():
            if layer_name in target_layer_data:
                # Call the comparison visualization function
                self._visualize_source_target_layer_comparison(
                    source_layer_data[layer_name],
                    source_labels,
                    target_layer_data[layer_name],
                    target_labels,
                    layer_name,
                    final_epoch
                )

    def visualize_layer_adaptation(self):
        """
        Visualize how much each layer has adapted during transfer learning
        """
        print("\nVisualizing layer adaptation during transfer learning...")

        # Get final adaptation metrics
        if not self.adaptation_metrics['layer_distances']:
            print("No adaptation metrics available")
            return

        final_entry = self.adaptation_metrics['layer_distances'][-1]

        # Get layer distances
        layer_distances = {}
        for name, distance in final_entry['distances'].items():
            layer_distances[name] = distance

        # Create visualization
        self._visualize_layer_adaptation_chart(layer_distances, final_entry['epoch'])

    def _visualize_layer_adaptation_chart(self, layer_distances, epoch):
        """
        Create chart showing how much each layer adapted during transfer
        """
        # Extract layer names and distances
        layer_names = []
        distances = []

        # Group parameters by layer
        layer_groups = {}
        for name, distance in layer_distances.items():
            # Extract layer number from parameter name
            if 'layers' in name:
                # Extract layer number assuming format 'layers.X.weight'
                parts = name.split('.')
                if len(parts) >= 2:
                    layer_num = int(parts[1]) // 3  # Convert to logical layer number
                    friendly_name = f"Layer {layer_num}"
                    if friendly_name not in layer_groups:
                        layer_groups[friendly_name] = []
                    layer_groups[friendly_name].append(distance)

        # Calculate average distance per layer
        for layer, distances_list in sorted(layer_groups.items()):
            layer_names.append(layer)
            distances.append(np.mean(distances_list))

        # Sort by distance for better visualization
        pairs = sorted(zip(layer_names, distances), key=lambda x: x[1], reverse=True)
        layer_names = [p[0] for p in pairs]
        distances = [p[1] for p in pairs]

        # Create horizontal bar chart
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(layer_names)), distances, align='center')
        plt.yticks(range(len(layer_names)), layer_names)
        plt.xlabel('Adaptation Distance (Parameter Change)')
        plt.title(f'Layer Adaptation: {self.source_dataset_name} → {self.target_dataset_name} (Epoch {epoch})')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Save figure
        filename = f'adaptation_{self.source_dataset_name}_to_{self.target_dataset_name}_epoch_{epoch}.png'
        plt.savefig(self.target_dir / filename)
        plt.close()

    # Add this to the TransferLearningManager class
    def prepare_model_for_target_dataset(self, input_size=3072):
        """
        Modify the model architecture to handle a new input size when transferring
        from a source dataset with different dimensions.

        This creates a new first layer that can handle the target dataset dimensions
        while keeping the weights of all other layers.

        Args:
            input_size: Input size for the target dataset (default: 3072 for SVHN)
        """
        print(f"Adapting model for target dataset with input size {input_size}...")

        # Get the current state dict to preserve weights for other layers
        state_dict = self.model.state_dict()

        # Get the output size of the first layer (input size of second layer)
        first_layer_output_size = self.model.layers[0].out_features

        # Create a new first layer with the correct input size
        new_first_layer = nn.Linear(input_size, first_layer_output_size)

        # Initialize the new first layer with appropriate weights
        nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(new_first_layer.bias, 0)

        # Replace the first layer in the model
        self.model.layers[0] = new_first_layer

        # Load the saved weights for all other layers
        model_dict = self.model.state_dict()

        # Create a filtered state dict with only the keys that match dimensions
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.size() == model_dict[k].size()}

        # Update model with preserved weights
        model_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_dict)

        print("Model adapted successfully.")


# Initialize the model
model = MLP().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Helper function to get activations for visualization
def get_layer_activations(model, data_loader, num_samples=2000):
    """
    Extract Final Layer Activations
    ---------------------------

    This utility function captures activations from the last hidden layer of the network
    for a specified number of samples. It's primarily used for analysis of the final
    learned representations before classification.

    The function puts the model in evaluation mode, processes batches of data until reaching the sample limit,
    extracts activations from the final ReLU layer and then returns both activations and corresponding labels.

    The last layer activations are particularly important as they represent
    the network's final feature representation before classification, showing
    how the network ultimately "sees" the data after all transformations.

    Args:
        model: Neural network model to analyze
        data_loader: DataLoader providing input samples
        num_samples (int): Number of samples to process (default: 2000)

    Returns:
        tuple: (activations, labels) as numpy arrays

    Raises:
        ValueError: If no activations could be collected
    """
    model.eval()
    activations = []
    labels = []
    count = 0

    # Get the key for the last ReLU layer
    last_relu_key = f'relu_{model.num_relu_layers}'

    with torch.no_grad():
        for data, target in data_loader:
            if count >= num_samples:
                break

            data = data.to(device)
            output = model(data)

            remaining = num_samples - count
            batch_size = min(data.size(0), remaining)

            if model.layer_activations[last_relu_key]:
                activation_batch = model.layer_activations[last_relu_key][-1][:batch_size]
                activations.append(activation_batch.cpu().numpy())
                labels.append(target[:batch_size].numpy())
                count += batch_size

            model.clear_activations()

    if not activations:
        raise ValueError(f"No activations were collected for layer {last_relu_key}")

    return np.vstack(activations), np.hstack(labels)


# Function to visualize the learned representations
def visualize_activations(activations, labels, title="t-SNE visualization of layer activations"):
    """
    Create t-SNE Visualization of Network Representations
    ------------------------------------------------

    This function creates a 2D visualization of high-dimensional activation data using t-SNE.
    The visualization shows how the network organizes data in its internal representation space.

    The visualization process projects high-dimensional activations to 2D using t-SNE,
    creates a scatter plot with points colored by class, adds a colorbar for class identification and
    labels the plot with a descriptive title.

    This visualization helps identify:
    - Cluster formation in the representation space
    - Class separation and decision boundaries
    - Potential misclassifications or confusion regions
    - Overall organization of the learned feature space

    Args:
        activations: High-dimensional activation data (numpy array)
        labels: Class labels corresponding to activations
        title (str): Plot title (default: "t-SNE visualization of layer activations")
    """
    tsne = TSNE(n_components=2, random_state=42)
    activations_2d = tsne.fit_transform(activations)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(activations_2d[:, 0], activations_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def get_friendly_layer_name(technical_name):
    """
    Convert technical layer names to human-readable descriptions.

    For MLP:
    - layers.1 → First Hidden Layer (ReLU)
    - layers.4 → Second Hidden Layer (ReLU)
    - layers.7 → Third Hidden Layer (ReLU)
    - layers.10 → Fourth Hidden Layer (ReLU)

    For CNN:
    - conv_block1.1 → First Conv Block (ReLU1)
    - conv_block1.3 → First Conv Block (ReLU2)
    - conv_block2.1 → Second Conv Block (ReLU1)
    - conv_block2.3 → Second Conv Block (ReLU2)
    - fc_layers.1 → First FC Layer (ReLU)
    - fc_layers.4 → Second FC Layer (ReLU)

    Args:
        technical_name: The original layer name

    Returns:
        str: Human-readable layer name
    """
    # MLP layer naming
    mlp_layer_names = {
        'relu_1': 'First Hidden Layer (ReLU)',
        'relu_2': 'Second Hidden Layer (ReLU)',
        'relu_3': 'Third Hidden Layer (ReLU)',
        'relu_4': 'Fourth Hidden Layer (ReLU)',
        # Legacy naming format
        'layers.1': 'First Hidden Layer (ReLU)',
        'layers.4': 'Second Hidden Layer (ReLU)',
        'layers.7': 'Third Hidden Layer (ReLU)',
        'layers.10': 'Fourth Hidden Layer (ReLU)'
    }

    # CNN layer naming
    cnn_layer_names = {
        'conv_block1.1': 'First Conv Block (ReLU1)',
        'conv_block1.3': 'First Conv Block (ReLU2)',
        'conv_block2.1': 'Second Conv Block (ReLU1)',
        'conv_block2.3': 'Second Conv Block (ReLU2)',
        'fc_layers.1': 'First FC Layer (ReLU)',
        'fc_layers.4': 'Second FC Layer (ReLU)'
    }

    # Combine dictionaries
    all_layer_names = {**mlp_layer_names, **cnn_layer_names}

    # Return the friendly name if available, otherwise keep the original
    return all_layer_names.get(technical_name, technical_name)


def create_comprehensive_analysis(trainer, test_loader, dataset_name="", phase=""):
    """
    Generate Complete Network Analysis Visualization
    -------------------------------------------

    This function creates a multipart analysis of the network's internal representations
    and behavior. It combines layer-wise visualizations with statistical analysis
    to provide a comprehensive understanding of how the network processes information.

    The analysis includes visualizations of how each layer represents the same input data,
    statistical metrics for each layer (mean activation, class separation) and
    comparative plots showing how metrics change across network depths

    This holistic view helps identify:
    - The most effective layers for class separation
    - How representations evolve through the network
    - Potential bottlenecks or redundant layers
    - The network's hierarchical feature extraction process

    The analysis extends the visualization approaches in Rauber et al. (2017)
    with additional quantitative metrics and comparative views.

    Args:
        trainer: TrainingManager instance containing the trained model
        test_loader: DataLoader providing test samples for analysis
        dataset_name (str): Name of the dataset for saving/titling
        phase (str): "Source", "Target", or other descriptor of training phase
    """
    # Layer-wise visualization
    print(f"\nGenerating layer-wise visualization for {dataset_name} ({phase})...")
    trainer.layerwise_viz.visualize_layer_comparisons(
        test_loader, dataset_name=dataset_name, phase=phase
    )

    # Compute layer statistics
    print(f"\nComputing layer statistics for {dataset_name} ({phase})...")
    stats = trainer.layerwise_viz.compute_layer_statistics(test_loader)

    # Create a visualization of layer statistics
    plt.figure(figsize=(15, 6))

    # Plot mean activations and class separation for each layer
    layers = list(stats.keys())
    friendly_layers = [get_friendly_layer_name(l) for l in layers]
    means = [stats[l]['mean_activation'] for l in layers]
    seps = [stats[l]['class_separation'] for l in layers]

    plt.subplot(1, 2, 1)
    plt.plot(range(len(layers)), means, 'b-o', label='Mean Activation')
    plt.xlabel('Layer')
    plt.ylabel('Mean Activation')
    plt.title(f'Mean Activation by Layer ({dataset_name} {phase})')
    plt.xticks(range(len(layers)), friendly_layers, rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(layers)), seps, 'r-o', label='Class Separation')
    plt.xlabel('Layer')
    plt.ylabel('Class Separation')
    plt.title(f'Class Separation by Layer ({dataset_name} {phase})')
    plt.xticks(range(len(layers)), friendly_layers, rotation=45)

    plt.tight_layout()

    # Save figure if we have a dataset name
    if dataset_name:
        phase_str = f"{phase}_" if phase else ""
        plt.savefig(f'{dataset_name}_{phase_str}layer_statistics.png')

    plt.show()


print(f"Model architecture:")
print(model)

# Initialize training manager and start training
trainer = TrainingManager(model, criterion, optimizer, device)

# Main execution
if __name__ == '__main__':
    """
    Main Execution Flow
    ----------------

    This section initializes the model, training infrastructure, and executes
    the complete training and visualization process. It demonstrates how to
    use the implemented visualization techniques to analyze a neural network
    during and after training.

    The execution sequence:
    1. Initialize the MLP model and move it to the appropriate device
    2. Set up loss function (CrossEntropyLoss) and optimizer (SGD with momentum)
    3. Create a TrainingManager to handle the training process
    4. Execute training with periodic activation capturing
    5. Generate training progress visualization
    6. Visualize final layer activations
    7. Create comprehensive layer-wise analysis

    This serves as an example workflow for applying the visualization techniques
    developed by Rauber et al. (2017) to understand neural network behavior.
    """
    # Step 1: Define SVHN data loaders (MNIST loaders are already defined above)
    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    # Load SVHN dataset
    svhn_train_dataset = datasets.SVHN(
        root='./data',
        split='train',
        download=True,
        transform=svhn_transform
    )

    svhn_test_dataset = datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=svhn_transform
    )

    # Create data loaders for SVHN
    svhn_train_loader = DataLoader(
        svhn_train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True
    )

    svhn_test_loader = DataLoader(
        svhn_test_dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=True
    )

    # Step 2: Initialize the transfer learning manager with the model that's already defined
    transfer_manager = TransferLearningManager(
        model=model,  # Use the model already defined in the file
        criterion=criterion,  # Use the criterion already defined in the file
        optimizer=optimizer,  # Use the optimizer already defined in the file
        device=device,  # Use the device already defined in the file
        source_dataset_name='MNIST',
        target_dataset_name='SVHN'
    )

    # Step 3: Train or load source model
    print("Training source model on MNIST...")
    transfer_manager.train_source_model(
        train_loader,  # Using MNIST train_loader already defined
        test_loader,  # Using MNIST test_loader already defined
        epochs=30,  # Reduced epochs for faster training
        activation_capture_frequency=5
    )

    # Step 4: Fine-tune on SVHN
    # Create a new optimizer with adjusted learning rate for fine-tuning
    fine_tune_optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    transfer_manager.optimizer = fine_tune_optimizer

    # Define SVHN input size
    svhn_input_size = 32 * 32 * 3  # 3072 for 32x32 RGB images

    # Fine-tune with different strategies
    print("\nFine-tuning on SVHN with all layers trainable...")
    transfer_manager.fine_tune(
        svhn_train_loader,
        svhn_test_loader,
        epochs=20,
        layers_to_freeze=None,  # Fine-tune all layers
        activation_capture_frequency=2,
        target_input_size=svhn_input_size  # Pass the correct input size
    )

    # For the second experiment, we need to:
    # 1. Create a completely new model with MNIST input size
    # 2. Initialize a new transfer manager
    # 3. Load the source model
    # 4. Then adapt and fine-tune

    print("\nPreparing for fine-tuning with frozen layers experiment...")

    # Initialize a new model with original input size
    new_model = MLP(input_size=784).to(device)

    # Create a new transfer manager
    new_transfer_manager = TransferLearningManager(
        model=new_model,
        criterion=criterion,
        optimizer=optimizer,  # Will be replaced before fine-tuning
        device=device,
        source_dataset_name='MNIST',
        target_dataset_name='SVHN',
        base_save_dir='./transfer_learning_frozen'  # Use a different directory
    )

    # Load the original source model (model still has MNIST input size)
    new_transfer_manager.load_source_model(Path('./transfer_learning/MNIST/source_model.pt'))

    # Create a new optimizer for fine-tuning with frozen layers
    frozen_fine_tune_optimizer = optim.SGD(new_model.parameters(), lr=0.0001, momentum=0.9)
    new_transfer_manager.optimizer = frozen_fine_tune_optimizer

    print("\nFine-tuning on SVHN with early layers frozen...")
    new_transfer_manager.fine_tune(
        svhn_train_loader,
        svhn_test_loader,
        epochs=20,
        layers_to_freeze=[0, 1],  # Freeze first two layers
        activation_capture_frequency=2,
        target_input_size=svhn_input_size
    )

    print("\nTransfer learning visualization complete.")
    print("All-layers fine-tuning visualizations saved in:", transfer_manager.target_dir)
    print("Frozen-layers fine-tuning visualizations saved in:", new_transfer_manager.target_dir)
