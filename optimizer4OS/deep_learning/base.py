import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
class BaseModel(nn.Module):
    def __init__(self, transform=None, folder_name="models", input_size=(28, 28)):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) if transform is None else transform
        self.optimizer = None
        self.scheduler = None
        self.folder_name = folder_name
        self.folder_path = None
        self.target_size = input_size
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def setup_folder(self, folder_name):
        """Create a folder if it doesn't exist."""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")
        return os.path.abspath(folder_name)

    def save_model(self, path):
        """Save the model to the specified path."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
            'device': self.device
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        """Load the model from the specified path."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)  # Ensure compatibility with different devices

        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load criterion state if available
        if 'criterion_state_dict' in checkpoint and self.criterion:
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])

        # Set device
        if 'device' in checkpoint:
            self.device = torch.device(checkpoint['device'])
        self.to(self.device)

        print(f"Model loaded from {path}")
        return checkpoint

    def accuracy(self, loader: DataLoader):
        """Calculate accuracy and validation loss."""
        val_running_loss = 0.0
        correct = 0
        total = 0
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = (correct / total) * 100
            return val_accuracy, val_running_loss / len(loader)

    def train_model(self, 
                    train_loader: DataLoader, 
                    val_loader: DataLoader, 
                    optimizer, 
                    scheduler=None,
                    criterion=None,
                    save_model=False,
                    epochs=10):
        """Train the model for the specified number of epochs."""
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else self.scheduler
        self.criterion = criterion if criterion is not None else self.criterion

        if save_model:
           self.learning_rate = self.optimizer.param_groups[0]['lr']
           self.folder_path = self.setup_folder(os.path.join(self.folder_name, str(self.learning_rate)))


        for epoch in range(epochs):
            self.train()  # Set model to training mode
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Calculate accuracy for this batch
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            # Store the average training loss and accuracy for the epoch
            self.train_losses.append(running_loss / len(train_loader))
            train_accuracy = (correct_train / total_train) * 100
            self.train_accuracies.append(train_accuracy)

            # Update the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Validate the model
            val_accuracy, val_loss = self.accuracy(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, val_loss {val_loss:.2f}")
            print(f"Train Accuracy: {train_accuracy:.2f}% Validation Accuracy: {val_accuracy:.2f}% ")

            if save_model:
                path = os.path.join(self.folder_path, f"model_train_{train_accuracy:.2f}_validation_{val_accuracy:.2f}_epoch_{epoch+1}.pth")
                self.save_model(path)
    
    def test_single_image(self, image_path):
        """Test a single image."""

        # Load the image
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        # Add a batch dimension (model expects [batch_size, channels, height, width])
        image = image.unsqueeze(0).to(self.device)

        # Set the model to evaluation mode and perform inference
        self.eval()
        with torch.no_grad():
            output = self(image)
            _, predicted_class = torch.max(output.data, 1)

        return predicted_class.item()# Set model to evaluation mode

    def process_input(self, input_data):
        """Process each input (URL, file path, numpy array, or PyTorch tensor) to return a tensor."""
        if isinstance(input_data, str):
            return self.process_image_from_path_or_url(input_data)
        elif isinstance(input_data, np.ndarray):
            return self.process_numpy_array(input_data)
        elif isinstance(input_data, torch.Tensor):
            return input_data.unsqueeze(0).to(self.device)
        else:
            raise ValueError("Unsupported input type. Must be URL, file path, numpy array, or PyTorch tensor.")

    def process_image_from_path_or_url(self, input_data):
        """Process an image from a URL or file path."""
        if input_data.startswith("http://") or input_data.startswith("https://"):
            response = requests.get(input_data)
            image = Image.open(BytesIO(response.content)).convert("L")
        else:
            image = Image.open(input_data).convert("L")  # Convert to grayscale
        return self.transform_image(image)

    def process_numpy_array(self, input_data):
        """Process a numpy array to a tensor."""
        if input_data.shape[-1] == 1:  # Single channel
            image = Image.fromarray(input_data.squeeze(axis=-1))
        else:
            raise ValueError("Numpy input must have a single channel.")
        return self.transform_image(image)

    def transform_image(self, image):
        """Apply transformations to the image and return a tensor."""
        if self.transform is None:
            raise ValueError("Transformation pipeline (self.transform) is not defined.")
        image = image.resize(self.target_size)
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, input_tensors):
        """Make predictions on input tensors."""
        with torch.no_grad():
            outputs = self(input_tensors)
            predicted_labels = torch.argmax(outputs, dim=1).cpu().tolist()
        return predicted_labels

    def test_multiple_inputs(self, input_data_list):
        """
        Test the model with multiple inputs (URLs, file paths, numpy arrays, or PyTorch tensors).

        Args:
            input_data_list: List of inputs to test. Can be URLs, file paths, numpy arrays, or PyTorch tensors.
            batch_process: If True, processes all inputs in a single batch; otherwise, processes one by one.

        Returns:
            A list of model outputs for each input.
        """
        self.eval()  # Set model to evaluation mode

        input_tensors = [self.process_input(input_data) for input_data in input_data_list]

        # Stack tensors for batch processing if batch_process is True
        input_tensors = torch.cat(input_tensors, dim=0)

        return self.predict(input_tensors)


    def plot_losses(self):
      """Plot training and validation loss and accuracy separately."""
      # Create a figure with two subplots
      fig, axes = plt.subplots(1, 2, figsize=(14, 5))

      # Plot training and validation loss
      axes[0].plot(self.train_losses, label='Training Loss', color='blue')
      axes[0].plot(self.val_losses, label='Validation Loss', color='orange')
      axes[0].set_title('Loss')
      axes[0].set_xlabel('Epochs')
      axes[0].set_ylabel('Loss')
      axes[0].legend()
      axes[0].grid(True)

      # Plot training and validation accuracy
      axes[1].plot(self.train_accuracies, label='Training Accuracy', color='green')
      axes[1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
      axes[1].set_title('Accuracy')
      axes[1].set_xlabel('Epochs')
      axes[1].set_ylabel('Accuracy (%)')
      axes[1].legend()
      axes[1].grid(True)

      # Add a tight layout to avoid overlapping labels
      plt.tight_layout()

      # Display the plots
      plt.show()