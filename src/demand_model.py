from typing import Dict, Optional, Union
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DemandModel:
    """
    Класс для моделирования спроса на основе логарифмической регрессии (Log-Log OLS).
    
    Моделирует степенную функцию спроса: Q(P) = A * P^beta
    В логарифмической форме: ln(Q) = alpha + beta * ln(P)
    
    Где:
        beta  - коэффициент ценовой эластичности спроса
        alpha - ln(A), базовый масштаб спроса
    """
    def __init__(self, min_samples: int = 5):
        """
        :param min_samples: Минимально необходимое число наблюдений для обучения модели.
        """
        self.min_samples = min_samples
        self.alfa: Optional[float] = None
        self.beta: Optional[float] = None
        self.smearing_factor: float = 1.0
        self.metrics: Dict[str, float] = {}
        self.ols_results: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self.is_fitted: bool = False