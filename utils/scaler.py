"""
Data scaling utilities for time series normalization
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class TimeSeriesScaler:
    """
    Handles scaling and inverse scaling for time series data
    """
    
    def __init__(self, scaler_type='minmax', feature_range=(0, 1)):
        """
        Initialize scaler
        
        Args:
            scaler_type: Type of scaler ('minmax' or 'standard')
            feature_range: Range for MinMaxScaler (default: (0, 1))
        """
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, data):
        """
        Fit scaler to data
        
        Args:
            data: numpy array of shape (n_samples, n_features)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self.scaler.fit(data)
        return self
    
    def transform(self, data):
        """
        Transform data using fitted scaler
        
        Args:
            data: numpy array to transform
            
        Returns:
            Scaled data
        """
        original_shape = data.shape
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        scaled = self.scaler.transform(data)
        
        # Restore original shape
        if len(original_shape) == 1:
            scaled = scaled.flatten()
        
        return scaled
    
    def fit_transform(self, data):
        """
        Fit and transform data
        
        Args:
            data: numpy array to fit and transform
            
        Returns:
            Scaled data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        Inverse transform scaled data back to original scale
        
        Args:
            data: Scaled data to inverse transform
            
        Returns:
            Data in original scale
        """
        original_shape = data.shape
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        inversed = self.scaler.inverse_transform(data)
        
        # Restore original shape
        if len(original_shape) == 1:
            inversed = inversed.flatten()
        
        return inversed
    
    def save(self, filepath):
        """
        Save scaler to file
        
        Args:
            filepath: Path to save scaler
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, filepath):
        """
        Load scaler from file
        
        Args:
            filepath: Path to load scaler from
        """
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        
        return self
    
    def get_params(self):
        """Get scaler parameters as JSON-serializable dict"""
        try:
            if self.scaler_type == 'minmax':
                params = {
                    'type': 'minmax',
                    'feature_range': list(self.feature_range)
                }
                # Add fitted parameters if scaler has been fitted
                if hasattr(self.scaler, 'data_min_'):
                    params['min'] = self.scaler.data_min_.tolist()
                    params['max'] = self.scaler.data_max_.tolist()
                return params
            else:
                params = {
                    'type': 'standard'
                }
                # Add fitted parameters if scaler has been fitted
                if hasattr(self.scaler, 'mean_'):
                    params['mean'] = self.scaler.mean_.tolist()
                    params['std'] = self.scaler.scale_.tolist()
                return params
        except Exception as e:
            # Fallback to basic info if conversion fails
            return {
                'type': self.scaler_type,
                'error': f'Could not serialize parameters: {str(e)}'
            }
