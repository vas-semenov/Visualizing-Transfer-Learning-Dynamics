import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from scipy.stats import pearsonr
import time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from keras import Sequential
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Dropout
from keras._tf_keras.keras.optimizers import Adam

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Fashion-MNIST label mapping
FASHION_MNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# MNIST label mapping (for reference)
MNIST_LABELS = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

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


class SSNP:
    """
    Self-Supervised Network Projection (SSNP) for supervised dimensionality reduction.
    Learns both direct (nD → 2D) and inverse (2D → nD) projections simultaneously.
    """

    def __init__(self, hidden_layers=3, hidden_size=256):
        self.direct_model = None
        self.inverse_model = None
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.is_fitted = False
        self.n_dims = None

    def build_models(self, n_dims):
        """Build the direct (nD → 2D) and inverse (2D → nD) projection models"""
        # Save dimensionality for later use
        self.n_dims = n_dims

        # Direct model: nD → 2D
        direct_model = Sequential()
        direct_model.add(Dense(self.hidden_size, input_dim=n_dims, activation='relu'))

        for _ in range(self.hidden_layers - 1):
            direct_model.add(BatchNormalization())
            direct_model.add(Dense(self.hidden_size, activation='relu'))
            direct_model.add(Dropout(0.2))

        direct_model.add(Dense(2))  # Output layer, no activation (linear)
        direct_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        # Inverse model: 2D → nD
        inverse_model = Sequential()
        inverse_model.add(Dense(self.hidden_size, input_dim=2, activation='relu'))

        for _ in range(self.hidden_layers - 1):
            inverse_model.add(BatchNormalization())
            inverse_model.add(Dense(self.hidden_size, activation='relu'))
            inverse_model.add(Dropout(0.2))

        inverse_model.add(Dense(n_dims))  # Output layer, no activation (linear)
        inverse_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        self.direct_model = direct_model
        self.inverse_model = inverse_model

    def fit(self, X, y=None, epochs=100, batch_size=32, verbose=0):
        """
        Train both direct and inverse projection models using labeled data.
        Makes use of labels to create better class separation.
        """
        from sklearn.preprocessing import MinMaxScaler

        # Scale the input data to [0,1] range
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build models if not already done
        if self.direct_model is None:
            self.build_models(X.shape[1])

        # Use t-SNE to create initial projections
        if y is not None:
            tsne = TSNE(n_components=2, random_state=42)
            projections = tsne.fit_transform(X_scaled)
        else:
            # Unsupervised projection
            tsne = TSNE(n_components=2, random_state=42)
            projections = tsne.fit_transform(X_scaled)

        # Scale projections to [0,1] range
        self.proj_scaler = MinMaxScaler()
        projections_scaled = self.proj_scaler.fit_transform(projections)

        # Train direct model: nD → 2D mapping
        self.direct_model.fit(X_scaled, projections_scaled,
                              epochs=epochs, batch_size=batch_size,
                              verbose=verbose)

        # Train inverse model: 2D → nD mapping
        self.inverse_model.fit(projections_scaled, X_scaled,
                               epochs=epochs, batch_size=batch_size,
                               verbose=verbose)

        self.is_fitted = True
        return self

    def transform(self, X):
        """Project high-dimensional data to 2D using the trained direct model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")

        # Scale the input data
        X_scaled = self.scaler.transform(X)

        # Apply direct projection
        projections_scaled = self.direct_model.predict(X_scaled)

        # Unscale projections
        return self.proj_scaler.inverse_transform(projections_scaled)

    def inverse_transform(self, projections):
        """Project 2D points back to high-dimensional space using inverse model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse transformation")

        # Scale the projection data
        proj_scaled = self.proj_scaler.transform(projections)

        # Apply inverse projection
        X_scaled = self.inverse_model.predict(proj_scaled)

        # Unscale the high-dimensional data
        return self.scaler.inverse_transform(X_scaled)


def create_sdbm_decision_boundary_map(model, train_loader, test_loader, epoch,
                                      dataset_name, phase, n_samples=2000,
                                      grid_size=100, boundary_alpha=0.5):
    """
    Creates a decision boundary visualization using the SDBM technique.
    This implementation follows the approach from the paper "SDBM: Supervised
    Decision Boundary Maps for Machine Learning Classifiers"
    """
    # Set model to evaluation mode
    model.eval()

    # 1. Collect data and labels for training the projections
    train_data = []
    train_labels = []

    with torch.no_grad():
        samples_collected = 0
        for data, labels in train_loader:
            if samples_collected >= n_samples:
                break

            batch_size = min(data.size(0), n_samples - samples_collected)
            data_batch = data[:batch_size].to(device)
            labels_batch = labels[:batch_size]

            # Extract features from the penultimate layer for better visualization
            features = get_penultimate_features(model, data_batch)

            # Store data
            train_data.append(features.cpu().numpy())
            train_labels.extend(labels_batch.cpu().numpy())

            samples_collected += batch_size

    # Combine batches
    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    # 2. Collect test data for evaluation
    test_data = []
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        samples_collected = 0
        for data, labels in test_loader:
            if samples_collected >= n_samples:
                break

            batch_size = min(data.size(0), n_samples - samples_collected)
            data_batch = data[:batch_size].to(device)
            labels_batch = labels[:batch_size]

            # Get model predictions
            outputs = model(data_batch)
            _, predictions = outputs.max(1)

            # Extract features
            features = get_penultimate_features(model, data_batch)

            # Store data
            test_data.append(features.cpu().numpy())
            test_labels.extend(labels_batch.cpu().numpy())
            test_predictions.extend(predictions.cpu().numpy())

            samples_collected += batch_size

    # Combine batches
    test_data = np.vstack(test_data)
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # 3. Train the SSNP model for supervised projection
    print(f"Training SSNP projection for {dataset_name} (Epoch {epoch})...")
    projection = SSNP(hidden_layers=3, hidden_size=256)
    projection.fit(train_data, train_labels, epochs=50, batch_size=64, verbose=0)

    # 4. Project test data to 2D
    test_projected = projection.transform(test_data)

    # 5. Create dense grid in 2D space
    x_min, x_max = test_projected[:, 0].min() - 1, test_projected[:, 0].max() + 1
    y_min, y_max = test_projected[:, 1].min() - 1, test_projected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 6. Map grid points back to high-dimensional space
    print(f"Generating synthetic points for decision boundary visualization...")
    synthetic_high_dim = projection.inverse_transform(grid_points)

    # 7. Classify synthetic points
    synthetic_predictions = []
    batch_size = 256  # Process in batches to avoid memory issues

    with torch.no_grad():
        for i in range(0, synthetic_high_dim.shape[0], batch_size):
            batch = synthetic_high_dim[i:i + batch_size]

            # Convert to tensor
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            # Pass through the classifier's fully connected layers directly
            # Since this is feature data from the penultimate layer
            outputs = model.fc_layers[-1](batch_tensor)
            _, predicted = outputs.max(1)

            synthetic_predictions.extend(predicted.cpu().numpy())

    # Reshape predictions to grid
    Z = np.array(synthetic_predictions).reshape(xx.shape)

    # 8. Plot the decision boundaries
    plt.figure(figsize=(10, 8))

    # Plot decision regions
    contour = plt.contourf(xx, yy, Z, alpha=boundary_alpha, cmap='tab10')

    # Add title and labels
    plt.title(f"Dataset: {dataset_name} | Phase: {phase} | Epoch: {epoch}")
    plt.xlabel("SDBM Dimension 1")
    plt.ylabel("SDBM Dimension 2")

    # Add color bar
    plt.colorbar(contour, label="Class")

    # Save the visualization
    plt.tight_layout()
    filename = f"{dataset_name.lower()}_{phase.lower().replace(' ', '_')}_epoch_{epoch}_sdbm_boundary.png"
    plt.savefig(filename)
    plt.close()

    print(f"SDBM decision boundary visualization saved as {filename}")


def get_penultimate_features(model, data):
    """
    Extract features from the penultimate layer for SDBM visualization
    """
    # Process through convolutional blocks
    x = model.conv_block1(data)
    x = model.conv_block2(x)
    x = x.view(x.size(0), -1)  # Flatten

    # Process through fully connected layers, except the last one
    for i in range(len(model.fc_layers) - 1):
        x = model.fc_layers[i](x)

    return x


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
        # Add these lines to store reference samples
        self.reference_data = {}
        self.reference_labels = {}

    def get_layer_representations(self, data_loader, num_samples=2000, memory_efficient=True):
        """
        Collect activations from all layers with improved memory management.

        Args:
            data_loader: DataLoader providing input samples
            num_samples: Number of samples to process
            memory_efficient: Whether to use memory-efficient processing

        Returns:
            tuple: (all_layers_data, labels)
        """
        self.model.eval()
        all_layers_data = {}
        labels = []
        count = 0

        # Process in smaller batches when in memory-efficient mode
        effective_batch_size = 16 if memory_efficient else None  # Limit batch processing size

        with torch.no_grad():
            for data, target in data_loader:
                if count >= num_samples:
                    break

                remaining = num_samples - count
                batch_size = min(data.size(0), remaining)
                if effective_batch_size:
                    batch_size = min(batch_size, effective_batch_size)

                # Process smaller chunks when in memory-efficient mode
                sub_batches = 1 if not memory_efficient else max(1, batch_size // effective_batch_size)

                for sub_idx in range(sub_batches):
                    if count >= num_samples:
                        break

                    # Calculate indices for this sub-batch
                    start_idx = sub_idx * effective_batch_size if memory_efficient else 0
                    end_idx = min(batch_size, start_idx + effective_batch_size) if memory_efficient else batch_size

                    # Forward pass to collect activations
                    sub_data = data[start_idx:end_idx].to(self.device)
                    _ = self.model(sub_data)

                    # Store activations from each layer
                    for layer_name, activations in self.model.layer_activations.items():
                        if layer_name not in all_layers_data:
                            all_layers_data[layer_name] = []

                        # Store activations, converting to numpy immediately to free GPU memory
                        layer_data = activations[:end_idx - start_idx].cpu().numpy().copy()
                        all_layers_data[layer_name].append(layer_data)

                    # Store labels
                    sub_labels = target[start_idx:end_idx].numpy().copy()
                    labels.extend(sub_labels)
                    count += end_idx - start_idx

                    # Clear activations after each sub-batch to free memory
                    self.model.clear_activations()

                    # Force garbage collection in memory-efficient mode
                    if memory_efficient and sub_idx % 4 == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        # Concatenate all batches for each layer
        for layer_name in all_layers_data:
            all_layers_data[layer_name] = np.vstack(all_layers_data[layer_name])

        return all_layers_data, np.array(labels)

    def visualize_layer_comparisons(self, data_loader, num_samples=2000,
                                    dataset_name="", phase="", epoch=None, save_fig=True,
                                    custom_labels=None):
        """
        Create comparative visualization of layer representations with dataset and epoch information.

        Args:
            data_loader: DataLoader providing input samples
            num_samples: Number of samples to visualize
            dataset_name: Name of the dataset (e.g., "MNIST", "Fashion-MNIST")
            phase: Training phase (e.g., "Source Training", "Transfer Learning")
            epoch: Current epoch number
            save_fig: Whether to save the figure to a file
            custom_labels: Dictionary mapping class indices to human-readable labels
        """
        # Get activations from all layers
        layer_data, labels = self.get_layer_representations(data_loader, num_samples)

        # Create subplot grid
        num_layers = len(layer_data)
        fig_size = min(20, num_layers * 5)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Set custom labels for colorbar if dataset is Fashion-MNIST
        if custom_labels is None and dataset_name == "Fashion-MNIST":
            custom_labels = FASHION_MNIST_LABELS

        # Create t-SNE projection for each layer
        for idx, (layer_name, activations) in enumerate(layer_data.items(), 1):
            # Get friendly layer name
            friendly_name = get_friendly_layer_name(layer_name)

            # If first time seeing this layer or not using consistent layout,
            # store current data as reference
            if layer_name not in self.reference_data or phase == "Source Training":
                self.reference_data[layer_name] = activations
                self.reference_labels[layer_name] = labels

                # Compute t-SNE projection with fixed random seed
                tsne = TSNE(n_components=2, random_state=42)
                projections = tsne.fit_transform(activations)
            else:
                # Use fixed random seed to maintain consistency
                tsne = TSNE(n_components=2, random_state=42)

                # Combine reference and current data
                combined_data = np.vstack([self.reference_data[layer_name], activations])
                combined_labels = np.concatenate([self.reference_labels[layer_name], labels])

                # Compute projection on combined data
                combined_projections = tsne.fit_transform(combined_data)

                # Extract only the new data points' projections
                projections = combined_projections[len(self.reference_data[layer_name]):]

            # Create subplot
            ax = fig.add_subplot(int(np.ceil(num_layers / 2)), 2, idx)
            scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                 c=labels, cmap='tab10', alpha=0.6)
            ax.set_title(f'{friendly_name}')

            if idx == 1:  # Add colorbar only for first plot
                if custom_labels:
                    # Create a colorbar with custom labels
                    cbar = plt.colorbar(scatter, ticks=range(len(custom_labels)))
                    cbar.set_label("Class")
                    # Set custom labels for colorbar
                    cbar.ax.set_yticklabels([custom_labels[i] for i in range(len(custom_labels))])
                else:
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


class CNNTransferLearningManager:
    """
    Handles the complete transfer learning pipeline for CNNs, including:
    1. Training/loading a source CNN model
    2. Capturing source model representations
    3. Fine-tuning on a target dataset
    4. Tracking representation changes during fine-tuning
    5. Visualizing the transfer learning dynamics
    """

    def __init__(self, model, criterion, optimizer, device,
                 source_dataset_name, target_dataset_name,
                 base_save_dir='./cnn_transfer_learning'):
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
        self.adaptation_metrics = {
            'layer_distances': [],
            'epoch_adaptation_speed': []
        }

        # Create visualization tools
        self.layer_viz = LayerwiseVisualization(model, device)

    def clear_memory(self):
        """
        Aggressively clear memory to prevent memory leaks and reduce fragmentation.
        Call this periodically during training.
        """
        # Clear CUDA cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Clear model activations
        self.model.clear_activations()

        # Clear visualization caches
        if hasattr(self.layer_viz, 'layer_activations'):
            self.layer_viz.layer_activations.clear()

        # Force garbage collection
        import gc
        gc.collect()

    def train_source_model(self, train_loader, test_loader, epochs=30,
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
            'timestamp': time.strftime("%Y%m%d-%H%M%S")
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
        print(f"Loaded source model trained on {checkpoint.get('dataset', 'unknown')}")

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

    def prepare_model_for_target_dataset(self, input_channels=3, reinit_conv_layers=[0]):
        """
        Modify the CNN architecture to handle a new input size when transferring
        from a source dataset with different dimensions.

        For CNNs, this typically means adjusting the first convolutional layer
        to handle a different number of input channels.

        Args:
            input_channels: Number of input channels for the target dataset
            reinit_conv_layers: List of convolutional blocks to reinitialize
        """
        print(f"Adapting model for target dataset with input channels {input_channels}...")
        print(f"Reinitializing convolutional blocks: {reinit_conv_layers}")

        # Get the current state dict to preserve weights for other layers
        state_dict = self.model.state_dict()

        # If first conv block needs reinitialization
        if 0 in reinit_conv_layers:
            # Get the first conv layer
            first_conv = self.model.conv_block1[0]

            # Create a new first layer with the correct input channels
            # but keep the same output channels and kernel size
            out_channels = first_conv.out_channels
            kernel_size = first_conv.kernel_size
            padding = first_conv.padding

            new_conv = nn.Conv2d(input_channels, out_channels,
                                 kernel_size, padding=padding)

            # Initialize with appropriate weights
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(new_conv.bias, 0)

            # Replace the first layer
            self.model.conv_block1[0] = new_conv
            print(f"  Reinitialized first conv layer: {input_channels} → {out_channels} channels")

        # Load the saved weights for all other layers
        model_dict = self.model.state_dict()

        # Create a filtered state dict with only the keys that match dimensions
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.size() == model_dict[k].size()}

        # Update model with preserved weights
        model_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_dict)

        print("Model adapted successfully.")

    def fine_tune(self, train_loader, test_loader, epochs=70,
                  activation_capture_frequency=10,
                  learning_rate_reduction=0.1, input_channels=3,
                  reinit_conv_layers=[0], lr_scheduler_type='cosine',
                  visualization_samples=2000, memory_optimization=True):
        """
        Fine-tune the model on the target dataset with careful monitoring
        of how representations change during transfer

        Args:
            train_loader: DataLoader for target training data
            test_loader: DataLoader for target test data
            epochs: Number of fine-tuning epochs
            activation_capture_frequency: How often to capture activations
            learning_rate_reduction: Factor to reduce learning rate (compared to source)
            input_channels: Number of input channels for target dataset
            reinit_conv_layers: List of conv blocks to reinitialize
            lr_scheduler_type: Type of learning rate scheduler
            visualization_samples: Number of samples to use for visualization
            memory_optimization: Whether to use memory optimization
        """
        print(f"Fine-tuning model for {self.target_dataset_name} with extended training...")

        # Adapt model for the target dataset
        self.prepare_model_for_target_dataset(input_channels, reinit_conv_layers)

        # Save initial model state for comparison
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Adjust learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            # Store original learning rate for scheduler reference
            original_lr = param_group['lr']
            # Apply reduction factor
            param_group['lr'] *= learning_rate_reduction
            print(f"Initial fine-tuning learning rate: {param_group['lr']:.6f}")

        # Setup learning rate scheduler based on specified type
        if lr_scheduler_type == 'step':
            # Step decay: reduce LR by factor of 0.1 every epochs/3 epochs
            step_size = max(5, epochs // 3)
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)
            print(f"Using StepLR scheduler with step size {step_size}")
        elif lr_scheduler_type == 'cosine':
            # Cosine annealing: smoothly reduce LR following a cosine curve
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
            print("Using CosineAnnealingLR scheduler")
        else:  # 'plateau'
            # Reduce on plateau: reduce LR when validation loss stops improving
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            print("Using ReduceLROnPlateau scheduler")

        # Storage for tracking adaptation
        self.target_activations = {}
        self.adaptation_metrics = {
            'layer_distances': [],
            'epoch_adaptation_speed': []
        }

        # Track best performance for early stopping consideration
        best_acc = 0.0
        best_epoch = 0

        # Initial memory clearing
        if memory_optimization:
            self.clear_memory()

        # Fine-tuning loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # Evaluation phase
            test_loss, test_acc = self._evaluate(test_loader)

            # Update learning rate scheduler
            if lr_scheduler_type == 'plateau':
                scheduler.step(test_loss)
            else:
                scheduler.step()

            # Track best performance
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

            # Clear memory every few epochs
            if memory_optimization and epoch % 3 == 0:
                self.clear_memory()

            # Capture activations periodically
            if epoch % activation_capture_frequency == 0 or epoch == epochs - 1:
                # Clear memory before visualization
                if memory_optimization:
                    self.clear_memory()

                # Capture current representations with reduced sample count
                layer_data, labels = self.layer_viz.get_layer_representations(
                    test_loader,
                    num_samples=visualization_samples,
                    memory_efficient=memory_optimization
                )

                # Store for later analysis
                self.target_activations[epoch] = {
                    'layer_data': layer_data,
                    'labels': labels
                }

                # Compute adaptation metrics
                self._compute_adaptation_metrics(epoch, initial_state)

                # Visualize the current state
                self._visualize_transfer_state(epoch, consistent_layout=True)

                create_sdbm_decision_boundary_map(
                    self.model,
                    train_loader,
                    test_loader,
                    epoch,
                    self.target_dataset_name,
                    "Transfer Learning",
                    n_samples=min(2000, visualization_samples),
                    grid_size=100
                )

                # Periodically add comparison visualization
                if epoch == 0 or epoch % (activation_capture_frequency * 2) == 0 or epoch == epochs - 1:
                    visualize_sdbm_source_target_boundaries(
                        self,
                        train_loader,  # Source dataset loader
                        test_loader,  # Target dataset loader
                        epoch,
                        n_samples=min(1500, visualization_samples),
                        grid_size=80
                    )

                # Clear memory after visualization
                if memory_optimization:
                    self.clear_memory()

            # Print epoch summary
            print(f'\nFine-tuning Epoch {epoch} Summary:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Current LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Best Acc: {best_acc:.4f} (Epoch {best_epoch})')

            # Save checkpoint
            self._save_fine_tuning_checkpoint(epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            })

        # Final memory clearing before visualizations
        if memory_optimization:
            self.clear_memory()

        # Final visualization and analysis
        self._create_final_visualizations()

        print(f"Fine-tuning complete. Best accuracy: {best_acc:.4f} at epoch {best_epoch}")

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

        for name, current_params in self.model.state_dict().items():
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

    def _visualize_transfer_state(self, epoch, consistent_layout=True):
        """
        Create visualizations to show the current state of transfer learning
        for a given epoch
        """
        if epoch not in self.target_activations:
            return

        # Generate layer-wise visualizations for current state
        print(f"\nVisualizing transfer learning state at epoch {epoch}...")

        # Get target activations
        target_data = self.target_activations[epoch]['layer_data']
        target_labels = self.target_activations[epoch]['labels']

        # Plot layer-wise representations for target data with dataset name and epoch
        self._visualize_layer_representations(
            target_data,
            target_labels,
            epoch,
            consistent_layout=consistent_layout
        )

        # If we have adaptation metrics, visualize them
        if self.adaptation_metrics['layer_distances']:
            self._visualize_adaptation_metrics(epoch)

    def _visualize_layer_representations(self, layer_data, labels, epoch, consistent_layout=True):
        """Create visualizations of layer representations with dataset and epoch information"""
        num_layers = len(layer_data)
        fig_size = min(20, num_layers * 5)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Set custom labels for Fashion-MNIST
        custom_labels = FASHION_MNIST_LABELS if self.target_dataset_name == "Fashion-MNIST" else None

        # Create t-SNE projection for each layer
        for idx, (layer_name, activations) in enumerate(layer_data.items(), 1):
            # Get friendly layer name
            friendly_name = get_friendly_layer_name(layer_name)

            # If first time seeing this layer or not using consistent layout,
            # compute projection directly
            if layer_name not in self.layer_viz.reference_data or not consistent_layout:
                self.layer_viz.reference_data[layer_name] = activations.copy()
                self.layer_viz.reference_labels[layer_name] = labels.copy()

                # Compute t-SNE projection with fixed random seed
                tsne = TSNE(n_components=2, random_state=42)
                projections = tsne.fit_transform(activations)
            else:
                # Use fixed random seed to maintain consistency
                tsne = TSNE(n_components=2, random_state=42)

                # Combine reference and current data
                combined_data = np.vstack([self.layer_viz.reference_data[layer_name], activations])
                combined_labels = np.concatenate([self.layer_viz.reference_labels[layer_name], labels])

                # Compute projection on combined data
                combined_projections = tsne.fit_transform(combined_data)

                # Extract only the new data points' projections
                projections = combined_projections[len(self.layer_viz.reference_data[layer_name]):]

            # Create subplot
            ax = fig.add_subplot(int(np.ceil(num_layers / 2)), 2, idx)
            scatter = ax.scatter(projections[:, 0], projections[:, 1],
                                 c=labels, cmap='tab10', alpha=0.6)
            ax.set_title(f'{friendly_name}')

            if idx == 1:  # Add colorbar only for first plot
                if custom_labels:
                    # Create a colorbar with custom labels
                    cbar = plt.colorbar(scatter, ticks=range(len(custom_labels)))
                    cbar.set_label("Class")
                    # Set custom labels for colorbar
                    cbar.ax.set_yticklabels([custom_labels[i] for i in range(len(custom_labels))])
                else:
                    plt.colorbar(scatter)

        # Include dataset name and epoch in title
        title = f"Dataset: {self.target_dataset_name} | Phase: Transfer Learning | Epoch: {epoch} | Layer-wise Representation Comparison"

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Save the figure
        filename = f'{self.target_dataset_name}_transfer_learning_epoch_{epoch}_layer_representations.png'
        plt.savefig(self.target_dir / filename)
        plt.close(fig)

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

        # Group parameters by layer type
        layer_groups = {
            'conv_block1': [],
            'conv_block2': [],
            'fc_layers': []
        }

        # Collect distances by layer group
        for name, distance in epoch_distances.items():
            if 'conv_block1' in name:
                layer_groups['conv_block1'].append(distance)
            elif 'conv_block2' in name:
                layer_groups['conv_block2'].append(distance)
            elif 'fc_layers' in name:
                layer_groups['fc_layers'].append(distance)

        # Calculate average distance per layer group
        layer_names = []
        avg_distances = []

        for group_name, distances in layer_groups.items():
            if distances:  # Only add if there are distances for this group
                layer_names.append(group_name)
                avg_distances.append(np.mean(distances))

        # Create bar chart
        plt.bar(layer_names, avg_distances)
        plt.xlabel('Network Component')
        plt.ylabel('Average Parameter Distance from Source')
        plt.title(f'Layer Adaptation: {self.source_dataset_name} → {self.target_dataset_name} (Epoch {epoch})')
        plt.xticks(rotation=45)

        # Save the figure
        plt.tight_layout()
        filename = f'adaptation_{self.source_dataset_name}_to_{self.target_dataset_name}_epoch_{epoch}.png'
        plt.savefig(self.target_dir / filename)
        plt.close()

    def _create_final_visualizations(self):
        """Create comprehensive visualizations of the entire transfer process"""
        # Create adaptation trajectory visualization
        self._visualize_adaptation_trajectory()

        # Create comparison of source vs. final target representations
        self.visualize_transfer_comparison()

        # Create layer adaptation visualization
        self.visualize_layer_adaptation()

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
                friendly_name = get_friendly_layer_name(layer)
                plt.plot(epochs, changes, marker='o', label=friendly_name)

        plt.xlabel('Fine-tuning Epoch')
        plt.ylabel('Representation Change (Distance from Source)')
        plt.title('Layer Representation Evolution During Transfer Learning')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        filename = f'adaptation_trajectory_{self.source_dataset_name}_to_{self.target_dataset_name}.png'
        plt.savefig(self.target_dir / filename)
        plt.close()

    def visualize_transfer_comparison(self):
        """
        Create visualizations comparing source and target representations
        for each layer after transfer learning
        """
        print("\nCreating source vs target comparison visualizations...")

        # Get final target epoch
        final_epochs = sorted(list(self.target_activations.keys()))
        if not final_epochs:
            print("No target activations available for comparison")
            return

        final_epoch = final_epochs[-1]

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

        # Source representation (MNIST)
        plt.subplot(1, 2, 1)
        tsne = TSNE(n_components=2, random_state=42)
        source_proj = tsne.fit_transform(source_activations)

        source_scatter = plt.scatter(source_proj[:, 0], source_proj[:, 1],
                                     c=source_labels, cmap='tab10', alpha=0.6)
        plt.title(f'Source: {self.source_dataset_name} ({friendly_name})')

        # Use standard digit labels for MNIST (source)
        source_cbar = plt.colorbar(source_scatter, ticks=range(10))
        source_cbar.set_label("Class")
        source_cbar.ax.set_yticklabels([MNIST_LABELS[i] for i in range(10)])

        # Target representation (Fashion-MNIST)
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, random_state=42)
        target_proj = tsne.fit_transform(target_activations)

        target_scatter = plt.scatter(target_proj[:, 0], target_proj[:, 1],
                                     c=target_labels, cmap='tab10', alpha=0.6)
        plt.title(f'Target: {self.target_dataset_name} ({friendly_name}) - Epoch {target_epoch}')

        # Use Fashion-MNIST labels for target plot
        if self.target_dataset_name == "Fashion-MNIST":
            target_cbar = plt.colorbar(target_scatter, ticks=range(10))
            target_cbar.set_label("Class")
            target_cbar.ax.set_yticklabels([FASHION_MNIST_LABELS[i] for i in range(10)])
        else:
            plt.colorbar(target_scatter)

        plt.suptitle(f'Transfer Learning: {self.source_dataset_name} → {self.target_dataset_name} ({friendly_name})',
                     fontsize=16)
        plt.tight_layout()

        # Save figure
        filename = f'transfer_{self.source_dataset_name}_to_{self.target_dataset_name}_{layer_name}_epoch_{target_epoch}.png'
        plt.savefig(self.target_dir / filename)
        plt.close()

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

        # Create visualization
        plt.figure(figsize=(12, 6))

        # Group parameters by layer type
        layer_groups = {
            'conv_block1': [],
            'conv_block2': [],
            'fc_layers': []
        }

        # Collect distances by layer group
        for name, distance in final_entry['distances'].items():
            if 'conv_block1' in name:
                layer_groups['conv_block1'].append(distance)
            elif 'conv_block2' in name:
                layer_groups['conv_block2'].append(distance)
            elif 'fc_layers' in name:
                layer_groups['fc_layers'].append(distance)

        # Calculate average and std for each layer group
        layer_names = []
        avg_distances = []
        std_distances = []

        for group_name, distances in layer_groups.items():
            if distances:  # Only add if there are distances for this group
                layer_names.append(group_name)
                avg_distances.append(np.mean(distances))
                std_distances.append(np.std(distances))

        # Create bar chart with error bars
        plt.bar(layer_names, avg_distances, yerr=std_distances, capsize=10)
        plt.xlabel('Network Component')
        plt.ylabel('Parameter Change Distance')
        plt.title(f'Layer Adaptation: {self.source_dataset_name} → {self.target_dataset_name}')

        # Add horizontal line for average change
        avg_change = np.mean(avg_distances)
        plt.axhline(y=avg_change, color='r', linestyle='-',
                    label=f'Average Change: {avg_change:.4f}')
        plt.legend()

        # Save figure
        plt.tight_layout()
        filename = f'layer_adaptation_{self.source_dataset_name}_to_{self.target_dataset_name}.png'
        plt.savefig(self.target_dir / filename)
        plt.close()


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


def get_conv_features(model, data):
    """
    Extract features from a convolutional layer for visualization
    """
    # Run forward through conv blocks
    x = model.conv_block1(data)
    x = model.conv_block2(x)

    # Use global average pooling to get a fixed-size representation
    # This converts each feature map to a single value
    batch_size, channels, height, width = x.size()
    x = x.view(batch_size, channels, -1).mean(dim=2)

    return x


def visualize_sdbm_source_target_boundaries(transfer_manager, source_loader, target_loader, epoch,
                                            n_samples=1500, grid_size=80):
    """
    Creates a side-by-side comparison of decision boundaries for source and target datasets
    using the SDBM technique
    """
    model = transfer_manager.model
    source_name = transfer_manager.source_dataset_name
    target_name = transfer_manager.target_dataset_name

    # Set model to evaluation mode
    model.eval()

    plt.figure(figsize=(16, 8))

    # Collect all data for training the projection
    all_train_data = []
    all_train_labels = []

    # Process source and target datasets
    loaders = [(source_loader, source_name), (target_loader, target_name)]
    data_collections = []

    for loader, dataset_name in loaders:
        # Collect training data for projection
        train_data = []
        train_labels = []

        with torch.no_grad():
            samples_collected = 0
            for data, labels in loader:
                if samples_collected >= n_samples:
                    break

                batch_size = min(data.size(0), n_samples - samples_collected)
                data_batch = data[:batch_size].to(device)
                labels_batch = labels[:batch_size]

                # Extract features
                features = get_penultimate_features(model, data_batch)

                # Store data
                train_data.append(features.cpu().numpy())
                train_labels.extend(labels_batch.cpu().numpy())

                samples_collected += batch_size

        # Combine batches
        train_data = np.vstack(train_data)
        train_labels = np.array(train_labels)

        # Add to collections
        all_train_data.append(train_data)
        all_train_labels.append(train_labels)
        data_collections.append((train_data, train_labels))

    # Train a single SSNP model on combined data to ensure comparable projections
    combined_data = np.vstack(all_train_data)
    combined_labels = np.concatenate(all_train_labels)

    print(f"Training shared SSNP projection for comparison visualization...")
    projection = SSNP(hidden_layers=3, hidden_size=256)
    projection.fit(combined_data, combined_labels, epochs=50, batch_size=64, verbose=0)

    # Process each dataset with the shared projection
    for idx, ((data, labels), (loader_name, dataset_name)) in enumerate(zip(data_collections, loaders)):
        plt.subplot(1, 2, idx + 1)

        # Project data
        embedded_data = projection.transform(data)

        # Create grid for decision boundaries
        x_min, x_max = embedded_data[:, 0].min() - 1, embedded_data[:, 0].max() + 1
        y_min, y_max = embedded_data[:, 1].min() - 1, embedded_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                             np.linspace(y_min, y_max, grid_size))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Generate synthetic high-dimensional points
        synthetic_high_dim = projection.inverse_transform(grid_points)

        # Classify synthetic points
        synthetic_predictions = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, synthetic_high_dim.shape[0], batch_size):
                batch = synthetic_high_dim[i:i + batch_size]

                # Convert to tensor
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

                # Classify
                outputs = model.fc_layers[-1](batch_tensor)
                _, predicted = outputs.max(1)

                synthetic_predictions.extend(predicted.cpu().numpy())

        # Reshape predictions to grid
        Z = np.array(synthetic_predictions).reshape(xx.shape)

        # Plot decision regions
        contour = plt.contourf(xx, yy, Z, alpha=0.6, cmap='tab10')

        # Set subplot title
        title = f"Source: {source_name}" if idx == 0 else f"Target: {target_name} (Epoch {epoch})"
        plt.title(title)

        # Add colorbar only to the second plot
        if idx == 1:
            contour_cbar = plt.colorbar(contour, ticks=range(10))
            contour_cbar.set_label("Class")

            # Use Fashion-MNIST labels for the target dataset (right side)
            if target_name == "Fashion-MNIST":
                contour_cbar.ax.set_yticklabels([FASHION_MNIST_LABELS[i] for i in range(10)])

    # Add overall title
    plt.suptitle(f"SDBM Decision Boundary Comparison: {source_name} → {target_name}", fontsize=16)
    plt.tight_layout()

    # Save visualization
    filename = f"sdbm_comparison_{source_name.lower()}_to_{target_name.lower()}_epoch_{epoch}.png"
    plt.savefig(filename)
    plt.close()

    print(f"SDBM comparison visualization saved as {filename}")


def get_friendly_layer_name(technical_name):
    """
    Convert technical layer names to human-readable descriptions.

    Args:
        technical_name: The original layer name

    Returns:
        str: Human-readable layer name
    """
    # CNN layer naming
    cnn_layer_names = {
        'conv_block1.1': 'Conv Block 1 - First ReLU',
        'conv_block1.3': 'Conv Block 1 - Second ReLU',
        'conv_block2.1': 'Conv Block 2 - First ReLU',
        'conv_block2.3': 'Conv Block 2 - Second ReLU',
        'fc_layers.1': 'FC Layer 1 (4096)',
        'fc_layers.4': 'FC Layer 2 (512)'
    }

    # Return the friendly name if available, otherwise keep the original
    return cnn_layer_names.get(technical_name, technical_name)


# Initialize model, criterion, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Model architecture:")
print(model)

# Initialize trainer
trainer = TrainingManager(model, criterion, optimizer, device)

if __name__ == '__main__':
    # Step 1: Define Fashion-MNIST data loaders (MNIST loaders are already defined above)
    fashion_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean and std
    ])

    # Load Fashion-MNIST dataset
    fashion_train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=fashion_transform
    )

    fashion_test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=fashion_transform
    )

    # Create data loaders for Fashion-MNIST
    fashion_train_loader = DataLoader(
        fashion_train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )

    fashion_test_loader = DataLoader(
        fashion_test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

    # Initialize model, criterion, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Step 2: Initialize the transfer learning manager
    transfer_manager = CNNTransferLearningManager(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        source_dataset_name='MNIST',
        target_dataset_name='Fashion-MNIST'
    )

    # Step 3: Train or load source model
    print("Training source model on MNIST...")
    transfer_manager.train_source_model(
        train_loader,  # Using MNIST train_loader
        test_loader,  # Using MNIST test_loader
        epochs=30,  # Shorter training as CNN learns faster
        activation_capture_frequency=10
    )

    # Option: Alternatively, you could load a previously trained model
    # transfer_manager.load_source_model()

    # Step 4: Fine-tune on Fashion-MNIST
    # Create a new optimizer with appropriate learning rate for fine-tuning
    fine_tune_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    transfer_manager.optimizer = fine_tune_optimizer

    # Fine-tune with memory optimization
    print("\nFine-tuning on Fashion-MNIST...")
    transfer_manager.fine_tune(
        fashion_train_loader,
        fashion_test_loader,
        epochs=40,
        activation_capture_frequency=5,
        input_channels=1,  # Both MNIST and Fashion-MNIST have 1 channel
        reinit_conv_layers=[0],  # Reinitialize first conv block
        lr_scheduler_type='cosine',
        visualization_samples=2000,
        memory_optimization=True
    )

    print("\nTransfer learning visualization complete.")
    print("Visualizations saved in:", transfer_manager.target_dir)
