import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.demand_model import DemandModel
from src.optimizer import ProfitOptimizer


# =====================================================================
# Фикстуры (Fixtures)
# =====================================================================
@pytest.fixture
def synthetic_elastic_data():
    """
    Генерирует датасет с известной эластичностью спроса:
    Q = exp(12.0) * P^(-1.8)
    """
    np.random.seed(42)
    prices = np.linspace(50.0, 150.0, 30)
    true_alpha = 12.0
    true_beta = -1.8

    # Добавляем небольшой контролируемый шум
    log_q = true_alpha + true_beta * np.log(prices) + np.random.normal(0, 0.02, len(prices))
    quantities = np.exp(log_q)

    df = pd.DataFrame({"Price": prices, "Quantity": quantities})
    return df, true_alpha, true_beta


@pytest.fixture
def fitted_model(synthetic_elastic_data):
    """Возвращает уже обученную модель спроса."""
    df, _, _ = synthetic_elastic_data
    model = DemandModel()
    model.fit(df)
    return model


# =====================================================================
# Тесты для DemandModel
# =====================================================================
class TestDemandModel:

    def test_fit_and_parameter_recovery(self, synthetic_elastic_data):
        """Проверяем, что модель восстанавливает истинные параметры с высокой точностью."""
        df, true_alpha, true_beta = synthetic_elastic_data
        model = DemandModel()
        model.fit(df)

        assert model.is_fitted is True
        assert model.beta is not None
        assert model.alpha is not None
        assert np.isclose(model.beta, true_beta, atol=0.1)
        assert np.isclose(model.alpha, true_alpha, atol=0.5)
        assert model.metrics["r2"] > 0.95
        assert "mape" in model.metrics
        assert "mae" in model.metrics

    def test_predict_scalar_and_array(self, fitted_model):
        """Проверка работы predict для отдельного числа и массива numpy."""
        # 1. Скаляр
        pred_single = fitted_model.predict(100.0)
        assert isinstance(pred_single, float)
        assert pred_single > 0

        # 2. Массив цен
        prices_arr = np.array([80.0, 100.0, 120.0])
        preds_arr = fitted_model.predict(prices_arr)
        assert isinstance(preds_arr, np.ndarray)
        assert len(preds_arr) == 3
        # Проверяем экономический закон: при росте цены спрос должен падать
        assert preds_arr[0] > preds_arr[1] > preds_arr[2]

    def test_predict_before_fit_raises_error(self):
        """Попытка предсказания до обучения должна вызывать RuntimeError."""
        model = DemandModel()
        with pytest.raises(RuntimeError, match="Модель еще не обучена"):
            model.predict(100.0)

    def test_invalid_data_inputs(self):
        """Проверка обработки некорректных входных данных."""
        model = DemandModel(min_samples=5)

        # Недостаточно строк
        short_df = pd.DataFrame({"Price": [10.0, 20.0], "Quantity": [100, 80]})
        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(short_df)

        # Отрицательные цены
        invalid_df = pd.DataFrame({
            "Price": [-10.0, -20.0, 0.0, 10.0, 20.0],
            "Quantity": [100, 80, 50, 40, 30]
        })
        with pytest.raises(ValueError, match="После фильтрации положительных значений"):
            model.fit(invalid_df)

    def test_elasticity_interpretation(self, fitted_model):
        """Проверка генерации понятного текстового описания эластичности."""
        text = fitted_model.get_elasticity_interpretation()
        assert "Эластичный спрос" in text
        assert fitted_model.beta is not None
        assert f"{abs(fitted_model.beta):.2f}" in text


# =====================================================================
# Тесты для ProfitOptimizer
# =====================================================================
class TestProfitOptimizer:

    def test_unfitted_model_raises_error(self):
        """Оптимизатор не должен принимать необученную модель."""
        unfitted_model = DemandModel()
        with pytest.raises(ValueError, match="необученная модель"):
            ProfitOptimizer(demand_model=unfitted_model)

    def test_optimization_within_bounds(self, fitted_model):
        """
        Проверяем, что найденная оптимальная цена лежит строго в пределах
        заданных бизнес-ограничений (мин. маржа и лимит изменения цены).
        """
        optimizer = ProfitOptimizer(demand_model=fitted_model)
        unit_cost = 40.0
        current_price = 100.0
        min_margin = 0.15      # Мин. цена: 40 * 1.15 = 46.0
        max_change = 0.20      # Диапазон: [80.0, 120.0]

        res = optimizer.optimize_price(
            unit_cost=unit_cost,
            current_price=current_price,
            min_margin=min_margin,
            max_price_change=max_change
        )

        assert "optimal_price" in res
        assert "profit_uplift_pct" in res

        # Проверяем нахождение внутри границ
        assert res["optimal_price"] >= res["lower_bound"]
        assert res["optimal_price"] <= res["upper_bound"]
        assert res["lower_bound"] == 80.0
        assert res["upper_bound"] == 120.0

    def test_profit_maximization_logic(self, fitted_model):
        """
        Проверяем, что прибыль при оптимальной цене не меньше прибыли при текущей цене.
        """
        optimizer = ProfitOptimizer(demand_model=fitted_model)
        res = optimizer.optimize_price(
            unit_cost=50.0,
            current_price=100.0,
            min_margin=0.10,
            max_price_change=0.30
        )

        # Ожидаемая прибыль должна быть >= текущей
        assert res["expected_profit"] >= res["current_profit"] - 1e-5
        assert res["profit_uplift_abs"] >= -1e-5

    def test_invalid_cost_and_price(self, fitted_model):
        """Проверка исключений при нулевой или отрицательной себестоимости/цене."""
        optimizer = ProfitOptimizer(demand_model=fitted_model)

        with pytest.raises(ValueError, match="строго больше нуля"):
            optimizer.optimize_price(unit_cost=-10.0, current_price=100.0)

        with pytest.raises(ValueError, match="строго больше нуля"):
            optimizer.optimize_price(unit_cost=50.0, current_price=0.0)