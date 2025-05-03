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

# Data preparation for SVHN with improved normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
])

# Load SVHN datasets
train_dataset = datasets.SVHN(
    root='./data',
    split='train',
    download=True,
    transform=transform
)

test_dataset = datasets.SVHN(
    root='./data',
    split='test',
    download=True,
    transform=transform
)

# Create data loaders with adjusted batch size
train_loader = DataLoader(
    train_dataset,
    batch_size=128,  # Increased batch size for better stability
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    pin_memory=True
)


class MLP(nn.Module):
    def __init__(self, input_size=3072):
        super(MLP, self).__init__()

        # Store activation outputs for visualization
        self.layer_activations = defaultdict(list)

        # Define the layers with batch normalization
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second hidden layer
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Third hidden layer
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Fourth hidden layer
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Output layer
            nn.Linear(1000, 10)
        )

        # Initialize weights using He initialization
        self._initialize_weights()

        # Register hooks for activation capturing
        self.activation_hooks = []
        self._register_hooks()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _register_hooks(self):
        def get_activation_hook(name):
            def hook(module, input, output):
                self.layer_activations[name].append(output.detach().cpu())

            return hook

        relu_count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                relu_count += 1
                hook = module.register_forward_hook(
                    get_activation_hook(f'relu_{relu_count}')
                )
                self.activation_hooks.append(hook)

        self.num_relu_layers = relu_count

    def clear_activations(self):
        self.layer_activations.clear()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class TrainingManager:
    def __init__(self, model, criterion, optimizer, device, save_dir='./checkpoints'):
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
        """Trains the model for one epoch and returns metrics"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for a progress bar
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
        """Evaluates the model on the test set"""
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
        """Captures and saves network activations"""
        activations, labels = get_layer_activations(self.model, loader)
        self.activation_history.append({
            'epoch': epoch,
            'activations': activations,
            'labels': labels
        })

    def train(self, train_loader, test_loader, epochs=100, activation_capture_frequency=10):
        """Main training loop with activation capturing and layer-wise visualization"""
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
                # Add layer-wise visualization for this epoch
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

        # Save final training history
        self.save_training_history()

        print("\nComputing neuron correlations...")
        self.visualization.compute_neuron_correlations(test_loader)

        # Create final visualizations
        print("\nCreating final evolution visualization...")
        self.visualization.visualize_epoch_evolution()

        print("\nCreating final neuron relationship visualization...")
        self.visualization.visualize_neuron_relationships()

        print("\nCreating final layer-wise comparison...")
        self.layerwise_viz.visualize_layer_comparisons(test_loader)

    def save_checkpoint(self, epoch):
        """Saves model checkpoint and training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')

    def save_training_history(self):
        """Saves training metrics history"""
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f)


class AdvancedVisualization:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.epoch_activations = {}
        self.neuron_correlations = None

    def capture_epoch_activations(self, epoch, data_loader, num_samples=2000):
        """Captures and stores activations for a specific epoch"""
        activations, labels = get_layer_activations(self.model, data_loader, num_samples)
        self.epoch_activations[epoch] = {
            'activations': activations,
            'labels': labels
        }

    def compute_neuron_correlations(self, data_loader, num_samples=2000):
        """Computes correlation matrix between neurons based on their activations"""
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

    def visualize_epoch_evolution(self, epochs_to_show=None, max_plots_per_figure=12):
        """Creates a visualization showing how representations evolve across epochs"""
        if epochs_to_show is None:
            epochs_to_show = sorted(self.epoch_activations.keys())

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
                ax.set_title(f'Epoch {epoch}')

                if idx == 1:  # Add colorbar only for first plot
                    plt.colorbar(scatter)

            plt.suptitle(f'Evolution of Representations (Epochs {batch_epochs[0]}-{batch_epochs[-1]})')
            plt.tight_layout()
            plt.show()

    def visualize_neuron_relationships(self):
        """Creates a visualization of neuron relationships using MDS"""
        if self.neuron_correlations is None:
            raise ValueError("Must compute neuron correlations first!")

        # Convert correlations to distances (1 - |correlation|)
        distances = 1 - np.abs(self.neuron_correlations)

        # Use MDS to create 2D embedding of neurons
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        neuron_positions = mds.fit_transform(distances)

        # Create visualization
        plt.figure(figsize=(12, 12))

        # Color neurons based on their average correlation with others
        avg_correlations = np.mean(np.abs(self.neuron_correlations), axis=1)

        scatter = plt.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
                              c=avg_correlations, cmap='viridis',
                              s=100, alpha=0.6)
        plt.colorbar(scatter, label='Average Correlation Strength')

        plt.title('Neuron Relationship Map')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.show()


class LayerwiseVisualization:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Dictionary to store activations from all layers
        self.layer_activations = {}
        # Register hooks for all ReLU layers
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for all ReLU layers to capture activations"""

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

    def visualize_layer_comparisons(self, data_loader, num_samples=2000):
        """Create comparative visualization of layer representations"""
        # Get activations from all layers
        layer_data, labels = self.get_layer_representations(data_loader, num_samples)

        # Create subplot grid
        num_layers = len(layer_data)
        fig_size = min(20, num_layers * 5)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Create t-SNE projection for each layer
        for idx, (layer_name, activations) in enumerate(layer_data.items(), 1):
            # Compute t-SNE projection
            tsne = TSNE(n_components=2, random_state=42)
            projections = tsne.fit_transform(activations)

            # Create subplot
            ax = fig.add_subplot(int(np.ceil(num_layers / 2)), 2, idx)
            scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                 c=labels, cmap='tab10', alpha=0.6)
            ax.set_title(f'Layer: {layer_name}')

            if idx == 1:  # Add colorbar only for first plot
                plt.colorbar(scatter)

        plt.suptitle('Layer-wise Representation Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()

    def compute_layer_statistics(self, data_loader, num_samples=2000):
        """Compute statistical measures for each layer's representations"""
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


def plot_training_progress(trainer):
    """Creates a visualization of training metrics over time"""
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


# Initialize model, criterion, and optimizer with adjusted parameters
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()

# Use Adam optimizer with lower learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)


# Modified helper function to handle potential numerical instabilities
def get_layer_activations(model, data_loader, num_samples=2000):
    model.eval()
    activations = []
    labels = []
    count = 0

    last_relu_key = f'relu_{model.num_relu_layers}'

    with torch.no_grad():
        for data, target in data_loader:
            if count >= num_samples:
                break

            data = data.to(device)
            _ = model(data)

            remaining = num_samples - count
            batch_size = min(data.size(0), remaining)

            if model.layer_activations[last_relu_key]:
                activation_batch = model.layer_activations[last_relu_key][-1][:batch_size]
                # Add small epsilon to prevent numerical instability
                activation_batch = activation_batch + 1e-10
                activations.append(activation_batch.cpu().numpy())
                labels.append(target[:batch_size].numpy())
                count += batch_size

            model.clear_activations()

    if not activations:
        raise ValueError(f"No activations collected for layer {last_relu_key}")

    return np.vstack(activations), np.hstack(labels)


def visualize_activations(activations, labels, title="t-SNE visualization of layer activations"):
    # Add small epsilon to prevent numerical instability
    activations = activations + 1e-10

    tsne = TSNE(n_components=2, random_state=42)
    activations_2d = tsne.fit_transform(activations)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(activations_2d[:, 0], activations_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def create_comprehensive_analysis(trainer, test_loader):
    print("\nGenerating layer-wise visualization...")
    trainer.layerwise_viz.visualize_layer_comparisons(test_loader)

    print("\nComputing layer statistics...")
    stats = trainer.layerwise_viz.compute_layer_statistics(test_loader)

    plt.figure(figsize=(15, 6))

    layers = list(stats.keys())
    means = [stats[l]['mean_activation'] for l in layers]
    seps = [stats[l]['class_separation'] for l in layers]

    plt.subplot(1, 2, 1)
    plt.plot(range(len(layers)), means, 'b-o', label='Mean Activation')
    plt.xlabel('Layer')
    plt.ylabel('Mean Activation')
    plt.title('Mean Activation by Layer')
    plt.xticks(range(len(layers)), layers, rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(layers)), seps, 'r-o', label='Class Separation')
    plt.xlabel('Layer')
    plt.ylabel('Class Separation')
    plt.title('Class Separation by Layer')
    plt.xticks(range(len(layers)), layers, rotation=45)

    plt.tight_layout()
    plt.show()


print(f"Model architecture:")
print(model)

# Initialize training manager
trainer = TrainingManager(model, criterion, optimizer, device)

if __name__ == '__main__':
    trainer.train(train_loader, test_loader, epochs=100, activation_capture_frequency=10)

    plot_training_progress(trainer)

    if trainer.activation_history:
        final_activations = trainer.activation_history[-1]
        visualize_activations(
            final_activations['activations'],
            final_activations['labels'],
            f"Layer Activations at Epoch {final_activations['epoch']}"
        )

    create_comprehensive_analysis(trainer, test_loader)
