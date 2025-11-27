"""
CNN Training and Evaluation Framework

A flexible training framework for PyTorch CNN models with support for
hyperparameter tuning, learning rate scheduling, and evaluation metrics.

Author: Aarush Chhiber
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class Trainer:
    """
    Training and evaluation framework for CNN models.
    
    Provides a complete training loop with:
        - Configurable hyperparameters
        - Learning rate scheduling
        - Training/validation metrics tracking
        - Model prediction and evaluation
    
    Attributes:
        model (nn.Module): The PyTorch model to train
        trainset: Training dataset
        testset: Test/validation dataset
        device (str): Device to train on ('cuda' or 'cpu')
        train_loss_per_epoch (list): Training loss history
        train_accuracy_per_epoch (list): Training accuracy history
        test_loss_per_epoch (list): Validation loss history
        test_accuracy_per_epoch (list): Validation accuracy history
    """
    
    def __init__(
        self,
        model,
        trainset,
        testset,
        num_epochs=5,
        batch_size=32,
        init_lr=1e-3,
        device="cpu",
    ):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): PyTorch model to train
            trainset: Training dataset
            testset: Test/validation dataset
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            init_lr (float): Initial learning rate
            device (str): Device for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.trainset = trainset
        self.testset = testset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.device = device

        # Metrics tracking
        self.train_loss_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.test_loss_per_epoch = []
        self.test_accuracy_per_epoch = []

    def tune(self):
        """
        Run training with tuned hyperparameters.
        
        Sets optimized hyperparameters based on experimentation
        and initiates training.
        """
        self.num_epochs = 10
        self.batch_size = 32
        self.init_lr = 2e-3
        self.train()

    def run_epoch(self, total, correct, running_loss, data_iterator, train=True):
        """
        Process one epoch of training or validation.
        
        Args:
            total (int): Running total of samples processed
            correct (int): Running count of correct predictions
            running_loss (float): Accumulated loss
            data_iterator: DataLoader iterator with progress bar
            train (bool): Whether this is a training epoch
            
        Returns:
            tuple: (total, correct, running_loss) updated values
        """
        for batch in data_iterator:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if train:
                self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            
            if train:
                loss.backward()
                self.optimizer.step()
            
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            running_loss += loss.item()
            
            if train:
                data_iterator.set_postfix(
                    loss=running_loss / (total // labels.size(0)), 
                    accuracy=correct / total
                )
        
        if train:
            self.scheduler.step()
        
        return total, correct, running_loss

    def train(self):
        """
        Execute the full training loop.
        
        Trains the model for the specified number of epochs,
        tracking metrics and validating after each epoch.
        """
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adamax(self.model.parameters(), lr=self.init_lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            total = 0
            correct = 0
            running_loss = 0
            
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
                total, correct, running_loss = self.run_epoch(
                    total, correct, running_loss, tepoch, train=True
                )

            self.train_loss_per_epoch.append(running_loss / len(trainloader))
            self.train_accuracy_per_epoch.append(correct / total)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                test_total = 0
                test_correct = 0
                test_loss = 0

                test_total, test_correct, test_loss = self.run_epoch(
                    test_total, test_correct, test_loss, testloader, train=False
                )

                print(
                    f"Epoch {epoch + 1}: Validation Loss: {test_loss / len(testloader):.2f}, "
                    f"Validation Accuracy: {test_correct / test_total:.3f}"
                )
                self.test_loss_per_epoch.append(test_loss / len(testloader))
                self.test_accuracy_per_epoch.append(test_correct / test_total)

    def get_training_history(self):
        """
        Retrieve training history metrics.
        
        Returns:
            tuple: (train_loss, train_accuracy, test_loss, test_accuracy)
                   lists for each epoch
        """
        return (
            self.train_loss_per_epoch,
            self.train_accuracy_per_epoch,
            self.test_loss_per_epoch,
            self.test_accuracy_per_epoch,
        )

    def predict(self, testloader):
        """
        Generate predictions on test data.
        
        Args:
            testloader: DataLoader for test data
            
        Returns:
            tuple: (probabilities, predictions, ground_truth) as tensors
        """
        self.model.eval()
        predict_probs = []
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                predict_probs.append(F.softmax(outputs, dim=1))
                predictions.append(outputs.argmax(dim=1))
                ground_truth.append(labels)

        return (
            torch.cat(predict_probs).cpu(),
            torch.cat(predictions).cpu(),
            torch.cat(ground_truth).cpu(),
        )
