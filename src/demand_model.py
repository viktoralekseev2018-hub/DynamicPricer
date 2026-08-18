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
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.smearing_factor: float = 1.0
        self.metrics: Dict[str, float] = {}
        self.ols_results: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self.is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        price_col: str = "Price",
        quantity_col: str = "Quantity"
    ) -> "DemandModel":
        """
        Обучает Log-Log модель на агрегированных парах (цена, объем продаж).
        
        :param data: DataFrame с данными по товару.
        :param price_col: Название колонки с ценой.
        :param quantity_col: Название колонки с объемом спроса.
        :return: Обученный экземпляр DemandModel (self).
        """
        if len(data) < self.min_samples:
            raise ValueError(
                f"Недостаточно данных для обучения. Требуется минимум {self.min_samples} точек, передано {len(data)}."
            )

        # Проверка на неотрицательность
        valid_mask = (data[price_col] > 0) & (data[quantity_col] > 0)
        df_valid = data[valid_mask].copy()

        if len(df_valid) < self.min_samples:
            raise ValueError("После фильтрации положительных значений (P > 0, Q > 0) данных недостаточно.")

        # Безопасное извлечение одномерных numpy-массивов с явным типом float
        actual_prices = df_valid[price_col].to_numpy(dtype=float)
        actual_quantities = df_valid[quantity_col].to_numpy(dtype=float)

        # Логарифмирование признаков
        log_price = np.log(actual_prices)
        log_quantity = np.log(actual_quantities)

        # OLS регрессия с константой
        X = sm.add_constant(log_price)
        ols_model = sm.OLS(log_quantity, X)
        self.ols_results = ols_model.fit()

        self.alpha = float(self.ols_results.params[0])
        self.beta = float(self.ols_results.params[1])

        # Поправка Смики (Duan's Smearing Estimator) для несмещенного exp()
        residuals = self.ols_results.resid
        self.smearing_factor = float(np.mean(np.exp(residuals)))

        self.is_fitted = True

        # Предупреждение о нетипичной (положительной) эластичности
        if self.beta > 0:
            logger.warning(
                f"Получена положительная эластичность beta = {self.beta:.3f}. "
                "Это может свидетельствовать о товаре Веблена/Гиффена или зашумленности данных."
            )

        # Расчет метрик качества на обучающей выборке
        self._calculate_metrics(
            actual_prices=actual_prices,
            actual_quantities=actual_quantities
        )

        logger.info(
            f"Модель обучена: alpha={self.alpha:.3f}, beta={self.beta:.3f}, "
            f"R2={self.metrics['r2']:.3f}, MAPE={self.metrics['mape']:.2%}"
        )
        return self

    def predict(self, price: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Прогнозирует объем спроса (Q) для заданной цены (P).
        
        :param price: Цена (число или массив чисел).
        :return: Спрогнозированное количество продаж.
        """
        if not self.is_fitted or self.alpha is None or self.beta is None:
            raise RuntimeError("Модель еще не обучена. Сначала вызовите .fit().")

        price_arr = np.asarray(price, dtype=float)
        if np.any(price_arr <= 0):
            raise ValueError("Цена должна быть строго больше нуля.")

        # Расчет Q = exp(alpha + beta * ln(P)) * smearing_factor
        log_pred = self.alpha + self.beta * np.log(price_arr)
        pred_q = np.exp(log_pred) * self.smearing_factor

        if np.ndim(price) == 0:
            return float(pred_q)
        return pred_q

    def _calculate_metrics(self, actual_prices: np.ndarray, actual_quantities: np.ndarray) -> None:
        """Внутренний расчет метрик R2, MAE, MAPE."""
        if self.beta is None or self.ols_results is None:
            raise RuntimeError("Параметры модели не инициализированы.")

        predictions = self.predict(actual_prices)

        self.metrics = {
            "r2": float(r2_score(actual_quantities, predictions)),
            "mae": float(mean_absolute_error(actual_quantities, predictions)),
            "mape": float(mean_absolute_percentage_error(actual_quantities, predictions)),
            "elasticity": self.beta,
            "p_value_beta": float(self.ols_results.pvalues[1]),
            "smearing_factor": self.smearing_factor
        }

    def get_elasticity_interpretation(self) -> str:
        """
        Возвращает текстовую интерпретацию полученного коэффициента эластичности.
        """
        if not self.is_fitted or self.beta is None:
            raise RuntimeError("Модель еще не обучена.")

        if self.beta > 0:
            return f"Аномальный спрос (beta = {self.beta:.2f} > 0): объем продаж растет вместе с ценой."
        elif abs(self.beta) > 1.0:
            return (
                f"Эластичный спрос (beta = {self.beta:.2f}): при росте цены на 1% "
                f"спрос падает на {abs(self.beta):.2f}%. Повышение цены снижает общую выручку."
            )
        elif abs(self.beta) < 1.0:
            return (
                f"Неэластичный спрос (beta = {self.beta:.2f}): при росте цены на 1% "
                f"спрос снижается всего на {abs(self.beta):.2f}%. Бизнес может осторожно повышать цены."
            )
        else:
            return "Единичная эластичность (beta = -1.0): изменение цены пропорционально меняет спрос."


if __name__ == "__main__":
    np.random.seed(42)
    sample_prices = np.array([100, 110, 120, 130, 140, 150, 160, 170], dtype=float)
    sample_quantities = np.exp(12 - 1.5 * np.log(sample_prices)) * np.random.normal(1.0, 0.05, len(sample_prices))

    test_df = pd.DataFrame({"Price": sample_prices, "Quantity": sample_quantities})

    model = DemandModel()
    model.fit(test_df)

    print("\n--- Результаты тестирования ---")
    print(f"Коэффициент эластичности: {model.beta:.3f}")
    print(f"Интерпретация: {model.get_elasticity_interpretation()}")
    print(f"Метрики: {model.metrics}")
    
    test_p = 125.0
    print(f"Прогноз спроса при цене {test_p} руб.: {model.predict(test_p):.1f} шт.")