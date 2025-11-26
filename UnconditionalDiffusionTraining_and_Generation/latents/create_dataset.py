import torch
import numpy as np
from einops import rearrange


class Cartesian_Dataset:
    """
    This class is for case 1 in the paper.
    """
    def __init__(self, load_checkpoint, train_val_split=0.9):
        checkpoint = torch.load(load_checkpoint)
        checkpoint_latent = checkpoint["hidden_states"]["latents"]
        try:
            latent_images = rearrange(checkpoint_latent, "(b t) d -> b t d", t=64)
        except:
            raise ValueError(
                f"Checkpoint_latent should be in shape (b t) d, but got {checkpoint_latent.shape}"
            )
        latent_images = latent_images.unsqueeze(1)
        assert len(latent_images.shape) == 4, f"Expected data shape (b, 1, h, w), but got {latent_images.shape}"
        
        self.latent_images = latent_images
        self.max = torch.max(latent_images)
        self.min = torch.min(latent_images)
        self.train_val_split = train_val_split
    
    def create_dataset(self):
        """
        Create a dataset by splitting the latent images into training and validation sets.
        """
        whole_dataset = self.latent_images
        train_data, val_data = self._split_dataset(whole_dataset)
        print(f"-- Training data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
        
        norm_train_data = self._normalize(train_data, self.min, self.max)
        norm_val_data = self._normalize(val_data, self.min, self.max)
        
        return norm_train_data, norm_val_data
    
    def _split_dataset(self, whole_dataset):
        """
        Split the dataset into training and validation sets.
        """
        num_samples = whole_dataset.shape[0]
        split_index = int(num_samples * self.train_val_split)
        
        train_data = whole_dataset[:split_index]
        val_data = whole_dataset[split_index:]
        
        return train_data, val_data
    
    def _normalize(self, data, minimum, maximum):
        return -1 + (data - minimum) * 2. / (maximum - minimum)


class Norway_Dataset:
    """
    This class is the same for case 2 and case 3 in the paper.
    """
    def __init__(self, load_checkpoint, train_val_split=0.9):
        checkpoint = torch.load(load_checkpoint)
        checkpoint_latent = checkpoint["hidden_states"]["latents"].half()
        try:
            latent_images = rearrange(checkpoint_latent, "(b t) d -> b t d", t=128)
        except:
            raise ValueError(
                f"Checkpoint_latent should be in shape (b t) d, but got {checkpoint_latent.shape}"
            )
        latent_images = latent_images.unsqueeze(1)
        assert len(latent_images.shape) == 4, f"Expected data shape (b, 1, h, w), but got {latent_images.shape}"
        print(f"-- Loaded latent images with shape: {latent_images.shape}")
        
        self.latent_images = latent_images
        self.max = torch.max(latent_images)
        self.min = torch.min(latent_images)
        self.train_val_split = train_val_split
    
    def create_dataset(self):
        """
        Create a dataset by splitting the latent images into training and validation sets.
        """
        whole_dataset = self.latent_images
        train_data, val_data = self._split_dataset(whole_dataset)
        print(f"-- Training data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
        
        norm_train_data = self._normalize(train_data, self.min, self.max)
        norm_val_data = self._normalize(val_data, self.min, self.max)
        
        return norm_train_data, norm_val_data
    
    def _split_dataset(self, whole_dataset):
        """
        Split the dataset into training and validation sets.
        """
        num_samples = whole_dataset.shape[0]
        split_index = int(num_samples * self.train_val_split)
        
        train_data = whole_dataset[:split_index]
        val_data = whole_dataset[split_index:]
        
        return train_data, val_data
    
    def _normalize(self, data, minimum, maximum):
        return -1 + (data - minimum) * 2. / (maximum - minimum)

