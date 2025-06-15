from abc import ABC

class Model(ABC):
    """
    Abstract base class for all models.
    """

    def __init__(self, name: str):
        self.name = name

    def fit(self, data):
        """
        Train the model with the provided data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, data):
        """
        Make predictions using the trained model.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    
class ARModel(Model):
    """
    Abstract base class for autoregressive models.
    """

    def __init__(self, name: str, lags: int):
        super().__init__(name)
        self.lags = lags

    def fit(self, data):
        """
        Fit the autoregressive model to the provided data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, data):
        """
        Predict future values using the fitted model.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    

class VolatilityModel(Model):
    """
    Abstract base class for volatility models.
    """

    def __init__(self, name: str, p: int, q: int):
        super().__init__(name)
        self.p = p
        self.q = q

    def fit(self, data):
        """
        Fit the volatility model to the provided data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, data):
        """
        Predict future volatility using the fitted model.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    

