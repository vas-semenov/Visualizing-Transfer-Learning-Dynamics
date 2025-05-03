import torch
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preparation with normalization for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load datasets
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
# Using batch size of 32 as specified in the paper for CNNs
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Store activation outputs for visualization
        self.layer_activations = defaultdict(list)

        # First convolutional block
        # Input: 28x28x1 -> Output: 28x28x32 (after padding)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 14x14x32
            nn.Dropout(0.25)
        )

        # Second convolutional block
        # Input: 14x14x32 -> Output: 14x14x64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 7x7x64
            nn.Dropout(0.25)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        # Register hooks for activation capturing
        self.activation_hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations from ReLU layers"""

        def get_activation_hook(name):
            def hook(module, input, output):
                # For convolutional layers, flatten the spatial dimensions
                if len(output.shape) == 4:
                    batch_size = output.size(0)
                    output = output.view(batch_size, -1)
                self.layer_activations[name] = output.detach().cpu()

            return hook

        # Register hooks for ReLU layers in both conv and fc blocks
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.activation_hooks.append(hook)

    def forward(self, x):
        # First convolutional block
        x = self.conv_block1(x)

        # Second convolutional block
        x = self.conv_block2(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x

    def clear_activations(self):
        """Clear stored activations to free memory"""
        self.layer_activations.clear()


class LayerwiseVisualization:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.layer_activations = {}

    def get_layer_representations(self, data_loader, num_samples=2000):
        """Collect activations from all layers for visualization"""
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
                for layer_name, activations in self.model.layer_activations.items():
                    if layer_name not in all_layers_data:
                        all_layers_data[layer_name] = []
                    # Store only the needed samples
                    layer_data = activations[:batch_size].numpy()
                    all_layers_data[layer_name].append(layer_data)

                labels.extend(target[:batch_size].numpy())
                count += batch_size
                self.model.clear_activations()

        # Concatenate all batches for each layer
        for layer_name in all_layers_data:
            all_layers_data[layer_name] = np.vstack(all_layers_data[layer_name])

        return all_layers_data, np.array(labels)

    def visualize_layer_comparisons(self, data_loader, num_samples=2000):
        """Create comparative visualization of layer representations"""
        layer_data, labels = self.get_layer_representations(data_loader, num_samples)

        # Create subplot grid
        num_layers = len(layer_data)
        fig_size = min(20, num_layers * 5)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Create t-SNE projection for each layer
        for idx, (layer_name, activations) in enumerate(layer_data.items(), 1):
            tsne = TSNE(n_components=2, random_state=42)
            projections = tsne.fit_transform(activations)

            ax = fig.add_subplot(int(np.ceil(num_layers / 2)), 2, idx)
            scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                 c=labels, cmap='tab10', alpha=0.6)
            ax.set_title(f'Layer: {layer_name}')

            if idx == 1:
                plt.colorbar(scatter)

        plt.suptitle('Layer-wise Representation Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()

    def compute_layer_statistics(self, data_loader, num_samples=2000):
        """Compute statistical measures for each layer's representations"""
        layer_data, labels = self.get_layer_representations(data_loader, num_samples)
        stats = {}

        for layer_name, activations in layer_data.items():
            stats[layer_name] = {
                'mean_activation': np.mean(activations),
                'std_activation': np.std(activations),
                'sparsity': np.mean(activations == 0),
                'max_activation': np.max(activations),
                'min_activation': np.min(activations)
            }

            # Compute class separation metric
            class_means = []
            for class_idx in np.unique(labels):
                class_data = activations[labels == class_idx]
                class_means.append(np.mean(class_data, axis=0))

            class_means = np.array(class_means)
            distances = []
            for i in range(len(class_means)):
                for j in range(i + 1, len(class_means)):
                    dist = np.linalg.norm(class_means[i] - class_means[j])
                    distances.append(dist)

            stats[layer_name]['class_separation'] = np.mean(distances)

        return stats


class NeuronMapVisualization:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.neuron_correlations = None
        self.visualization_outputs = []

    def compute_neuron_correlations(self, data_loader, num_samples=2000, max_neurons=1000):
        """Computes correlation matrix for the last layer before classification"""
        self.model.eval()
        activations = []
        count = 0

        with torch.no_grad():
            for data, _ in data_loader:
                if count >= num_samples:
                    break

                data = data.to(self.device)
                remaining = num_samples - count
                batch_size = min(data.size(0), remaining)

                # Forward pass to get activations
                _ = self.model(data)

                # Get activations from the last ReLU layer (fc_layers.1)
                activation_batch = self.model.layer_activations['fc_layers.4'][:batch_size]
                activations.append(activation_batch.cpu().numpy())
                count += batch_size

                self.model.clear_activations()

        # Concatenate all batches
        activations = np.vstack(activations)

        # Sample neurons if there are too many
        num_neurons = activations.shape[1]
        if num_neurons > max_neurons:
            selected_neurons = np.random.choice(num_neurons, max_neurons, replace=False)
            activations = activations[:, selected_neurons]

        # Compute correlation matrix
        num_neurons = activations.shape[1]
        correlation_matrix = np.zeros((num_neurons, num_neurons))

        for i in range(num_neurons):
            for j in range(i + 1):  # Only compute lower triangle
                if np.std(activations[:, i]) == 0 or np.std(activations[:, j]) == 0:
                    correlation = 0
                else:
                    correlation, _ = pearsonr(activations[:, i], activations[:, j])
                    correlation = 0 if np.isnan(correlation) else correlation

                # Make matrix symmetric
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

        # Convert correlations to distances
        distances = 1 - np.abs(correlation_matrix)

        self.neuron_correlations = distances
        return distances

    def visualize_neuron_map(self):
        """Creates visualization of neuron relationships using MDS"""
        if self.neuron_correlations is None:
            raise ValueError("Must compute neuron correlations first!")

        # Use MDS to create 2D embedding of neurons
        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, n_init=1, max_iter=100)
        neuron_positions = mds.fit_transform(self.neuron_correlations)

        # Create visualization
        plt.figure(figsize=(15, 15))

        # Color neurons based on their average correlation with others
        avg_correlations = np.mean(np.abs(self.neuron_correlations), axis=1)

        scatter = plt.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
                              c=avg_correlations, cmap='viridis',
                              s=100, alpha=0.6)
        plt.colorbar(scatter, label='Average Correlation Strength')

        plt.title('CNN Neuron Relationship Map (Final Layer)')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.show()


class TrainingManager:
    def __init__(self, model, criterion, optimizer, device, save_dir='./cnn_checkpoints'):
        """
        Initialize the training manager with all necessary components for training and visualization.

        Args:
            model: The neural network model to train
            criterion: The loss function
            optimizer: The optimization algorithm
            device: The device to run computations on (CPU/GPU)
            save_dir: Directory to save checkpoints and visualizations
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize storage for training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        # Initialize visualization components
        self.layerwise_viz = LayerwiseVisualization(model, device)

    def train_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader containing training data
            epoch: Current epoch number

        Returns:
            tuple: (average loss, accuracy) for this epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress visualization
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move data to appropriate device
            data, target = data.to(self.device), target.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Calculate running statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar with current metrics
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return running_loss / len(train_loader), correct / total

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader containing test data

        Returns:
            tuple: (average loss, accuracy) on test set
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

    def train(self, train_loader, test_loader, epochs=100, visualization_frequency=10):
        """
        Main training loop with periodic visualizations.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            epochs: Total number of training epochs
            visualization_frequency: How often to create visualizations
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

            # Periodic visualization (only at specified intervals)
            if epoch % visualization_frequency == 0:
                print(f"\nGenerating layer-wise visualization for epoch {epoch}...")
                self.layerwise_viz.visualize_layer_comparisons(test_loader)

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

        # Create final visualizations
        print("\nGenerating final visualizations...")

        # Plot training progress
        plot_training_progress(self)

        # Final layer-wise visualization
        self.layerwise_viz.visualize_layer_comparisons(test_loader)

        # Create neuron map with limited neurons for computational efficiency
        print("\nCreating neuron map visualization...")
        neuron_viz = NeuronMapVisualization(self.model, self.device)
        neuron_viz.compute_neuron_correlations(test_loader, max_neurons=500)
        neuron_viz.visualize_neuron_map()

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint and training state.

        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')


def plot_training_progress(trainer):
    """Visualize training metrics over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.plot(trainer.test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Losses')
    ax1.legend()

    # Plot accuracies
    ax2.plot(trainer.train_accuracies, label='Train Accuracy')
    ax2.plot(trainer.test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracies')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Initialize model, criterion, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Model architecture:")
print(model)

# Initialize trainer
trainer = TrainingManager(model, criterion, optimizer, device)

if __name__ == '__main__':
    # Start training
    trainer.train(train_loader, test_loader, epochs=100, visualization_frequency=10)

    # Plot final training progress
    plot_training_progress(trainer)
