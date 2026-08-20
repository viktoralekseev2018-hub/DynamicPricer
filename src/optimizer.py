from typing import TYPE_CHECKING, Any, Dict, Union
import logging
import numpy as np
from scipy.optimize import minimize_scalar

# Разделение для статического анализатора (Pylance) и автономного runtime
if TYPE_CHECKING:
    from src.demand_model import DemandModel
else:
    try:
        from src.demand_model import DemandModel
    except ImportError:
        class DemandModel:
            is_fitted: bool
            def predict(self, price: Any) -> Any: ...

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProfitOptimizer:
    """
    Класс для поиска оптимальной цены товара (Sweet Spot), 
    максимизирующей чистую валовую прибыль с учетом эластичности спроса.
    """

    def __init__(self, demand_model: DemandModel):
        """
        :param demand_model: Обученный экземпляр класса DemandModel.
        """
        self.model = demand_model
        
        if not getattr(self.model, "is_fitted", False):
            raise ValueError("В оптимизатор передана необученная модель DemandModel. Сначала вызовите .fit().")

    def _negative_profit(self, price: float, unit_cost: float) -> float:
        """
        Целевая функция для SciPy (минимизируем отрицательную прибыль).
        
        Profit = (P - Cost) * Q(P)
        """
        # Если цена ниже или равна себестоимости, возвращаем штраф (бесконечность)
        if price <= unit_cost:
            return float("inf")
            
        pred_qty = self.model.predict(price)
        expected_quantity = float(np.asarray(pred_qty).item() if np.ndim(pred_qty) > 0 else pred_qty)
        profit = (price - unit_cost) * expected_quantity
        
        return -float(profit)

    def optimize_price(
        self, 
        unit_cost: float, 
        current_price: float, 
        min_margin: float = 0.10, 
        max_price_change: float = 0.20
    ) -> Dict[str, float]:
        """
        Ищет оптимальную цену для максимизации прибыли с учетом бизнес-ограничений.
        
        :param unit_cost: Себестоимость единицы товара (закупочная цена + логистика).
        :param current_price: Текущая розничная цена товара.
        :param min_margin: Минимально допустимая наценка (например, 0.10 = 10% к себестоимости).
        :param max_price_change: Максимально допустимое изменение цены за один раз (например, 0.20 = +/-20%).
        :return: Словарь с результатами оптимизации (оптимальная цена, прогноз продаж, профит, uplift).
        """
        if unit_cost <= 0 or current_price <= 0:
            raise ValueError("Себестоимость и текущая цена должны быть строго больше нуля.")

        # 1. Расчет бизнес-ограничений (Boundaries)
        min_allowed_by_margin = unit_cost * (1.0 + min_margin)
        min_allowed_by_change = current_price * (1.0 - max_price_change)
        lower_bound = max(min_allowed_by_margin, min_allowed_by_change)
        upper_bound = current_price * (1.0 + max_price_change)

        # Обработка конфликта ограничений
        if lower_bound >= upper_bound:
            logger.warning(
                f"Конфликт ограничений: нижняя граница ({lower_bound:.2f}) >= верхней ({upper_bound:.2f}). "
                "Оптимизация невозможна, будет применена минимально разрешенная цена для спасения маржинальности."
            )
            optimal_price = lower_bound
            opt_profit = -self._negative_profit(optimal_price, unit_cost)
            
        else:
            # 2. Запуск оптимизатора SciPy
            result = minimize_scalar(
                self._negative_profit,
                args=(unit_cost,),
                bounds=(lower_bound, upper_bound),
                method="bounded"
            )
            
            if not result.success:
                logger.error(f"Сбой оптимизации: {result.message}")
                optimal_price = current_price
                opt_profit = -self._negative_profit(current_price, unit_cost)
            else:
                optimal_price = float(result.x)
                opt_profit = -float(result.fun)

        # 3. Расчет базовых показателей при текущей цене (Base Case)
        curr_qty_pred = self.model.predict(current_price)
        current_quantity = float(np.asarray(curr_qty_pred).item() if np.ndim(curr_qty_pred) > 0 else curr_qty_pred)
        current_profit = (current_price - unit_cost) * current_quantity

        # 4. Расчет эффекта внедрения (Uplift)
        opt_qty_pred = self.model.predict(optimal_price)
        optimal_quantity = float(np.asarray(opt_qty_pred).item() if np.ndim(opt_qty_pred) > 0 else opt_qty_pred)
        
        profit_uplift_abs = opt_profit - current_profit
        if current_profit > 0:
            profit_uplift_pct = profit_uplift_abs / current_profit
        else:
            profit_uplift_pct = 0.0

        logger.info(
            f"Оптимизация завершена. Текущая цена: {current_price:.2f} -> Оптимальная: {optimal_price:.2f} "
            f"(Изменение: {((optimal_price / current_price) - 1):.1%}). "
            f"Ожидаемый прирост прибыли: {profit_uplift_pct:.1%}"
        )

        return {
            "current_price": float(current_price),
            "current_profit": float(current_profit),
            "current_quantity": float(current_quantity),
            "optimal_price": float(optimal_price),
            "expected_profit": float(opt_profit),
            "expected_quantity": float(optimal_quantity),
            "profit_uplift_abs": float(profit_uplift_abs),
            "profit_uplift_pct": float(profit_uplift_pct),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }


if __name__ == "__main__":
    class DummyModel:
        is_fitted: bool = True
        def predict(self, price: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            price_arr = np.asarray(price, dtype=float)
            return (10**6) * (price_arr ** -1.8)

    dummy_model = DummyModel()
    optimizer = ProfitOptimizer(demand_model=dummy_model)  # type: ignore[arg-type]
    
    results = optimizer.optimize_price(
        unit_cost=60.0, 
        current_price=100.0, 
        min_margin=0.15,
        max_price_change=0.20
    )
    
    print("\n--- Результат оптимизации бизнес-кейса ---")
    print(f"Текущая цена:       {results['current_price']:.2f} руб.")
    print(f"Оптимальная цена:   {results['optimal_price']:.2f} руб.")
    print(f"Текущая прибыль:    {results['current_profit']:.2f} руб.")
    print(f"Ожидаемая прибыль:  {results['expected_profit']:.2f} руб.")
    print(f"Прирост (Uplift %): {results['profit_uplift_pct']:.2%}")
    print(f"Границы оптимизации:[{results['lower_bound']:.2f} - {results['upper_bound']:.2f}] руб.")