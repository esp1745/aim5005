
import numpy as np
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. 
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector.
        """
        x = self._check_is_array(x)
        
        if self.minimum is None or self.maximum is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        diff_max_min = self.maximum - self.minimum
        
        # Fixed Bug: Prevent division by zero
        diff_max_min[diff_max_min == 0] = 1  
        
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std_dev = None  # Standard deviation
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. 
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array"
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and standard deviation for standardization.
        """
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std_dev = x.std(axis=0, ddof=0)  # Population std deviation
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the dataset.
        """
        x = self._check_is_array(x)
        
        if self.mean is None or self.std_dev is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Fixed Bug: Prevent division by zero
        std_dev = np.where(self.std_dev == 0, 1, self.std_dev)
        
        return (x - self.mean) / std_dev

    def fit_transform(self, x: List) -> np.ndarray:
        """
        Fit to the data and return the transformed version.
        """
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
