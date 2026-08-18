import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем корень проекта в sys.path для корректного импорта модуля src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


@pytest.fixture
def sample_csv_path(tmp_path):
    """
    Фикстура для создания временного синтетического CSV-файла с транзакциями.
    Включает как валидные данные, так и возвраты, NaN и невалидные цены.
    """
    data = {
        "InvoiceDate": [
            "2026-01-05 10:00:00",  # Неделя 1, SKU_A (2 шт по 100)
            "2026-01-06 12:00:00",  # Неделя 1, SKU_A (3 шт по 120) -> средневзвешенная = (200+360)/5 = 112
            "2026-01-07 15:00:00",  # Неделя 1, SKU_B (10 шт по 50)
            "2026-01-15 11:00:00",  # Неделя 2, SKU_A (5 шт по 110)
            "2026-01-08 09:00:00",  # Невалидная строка: возврат (Quantity < 0)
            "2026-01-09 14:00:00",  # Невалидная строка: нулевая цена (Price <= 0)
            None,                   # Невалидная строка: пропущенная дата
            "2026-01-10 16:00:00",  # Невалидная строка: пропущенный SKU
        ],
        "StockCode": [
            "SKU_A",
            "SKU_A",
            "SKU_B",
            "SKU_A",
            "SKU_A",
            "SKU_A",
            "SKU_A",
            None
        ],
        "Price": [100.0, 120.0, 50.0, 110.0, 100.0, 0.0, 100.0, 100.0],
        "Quantity": [2, 3, 10, 5, -1, 5, 2, 2]
    }
    
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_transactions.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_load_data_success(sample_csv_path):
    """Проверка успешной загрузки данных и приведения типов."""
    loader = DataLoader(sample_csv_path)
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert "InvoiceDate" in df.columns
    assert "StockCode" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])


def test_load_data_file_not_found():
    """Проверка выброса исключения при отсутствии файла."""
    loader = DataLoader("non_existing_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_load_data_missing_columns(tmp_path):
    """Проверка ошибки, если в CSV отсутствуют обязательные столбцы."""
    bad_df = pd.DataFrame({"SomeColumn": [1, 2, 3]})
    bad_csv = tmp_path / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)
    
    loader = DataLoader(bad_csv)
    with pytest.raises(KeyError):
        loader.load_data()


def test_clean_data(sample_csv_path):
    """Проверка фильтрации невалидных данных (возвраты, NaN, нулевые цены)."""
    loader = DataLoader(sample_csv_path)
    loader.load_data()
    clean_df = loader.clean_data()
    
    # Из 8 строк валидными должны остаться только 4
    assert len(clean_df) == 4
    assert (clean_df["Quantity"] > 0).all()
    assert (clean_df["Price"] > 0).all()
    assert clean_df["StockCode"].isna().sum() == 0
    assert clean_df["InvoiceDate"].isna().sum() == 0


def test_get_aggregated_sku_data(sample_csv_path):
    """Проверка недельной агрегации и корректности средневзвешенной цены."""
    loader = DataLoader(sample_csv_path)
    loader.load_data()
    loader.clean_data()
    
    sku_a_weekly = loader.get_aggregated_sku_data("SKU_A", freq="W")
    
    # Для SKU_A 2 транзакции на первой неделе и 1 на второй неделе
    assert len(sku_a_weekly) == 2
    assert "Quantity" in sku_a_weekly.columns
    assert "Price" in sku_a_weekly.columns
    
    # Проверяем первую неделю: (2*100 + 3*120) / (2 + 3) = 560 / 5 = 112.0
    first_week = sku_a_weekly.iloc[0]
    assert first_week["Quantity"] == 5
    assert np.isclose(first_week["Price"], 112.0)


def test_get_aggregated_sku_not_found(sample_csv_path):
    """Проверка обработки запроса по несуществующему SKU."""
    loader = DataLoader(sample_csv_path)
    loader.load_data()
    loader.clean_data()
    
    with pytest.raises(ValueError, match="не найден"):
        loader.get_aggregated_sku_data("NON_EXISTING_SKU")


def test_get_top_skus(sample_csv_path):
    """Проверка получения топа популярных товаров."""
    loader = DataLoader(sample_csv_path)
    loader.load_data()
    loader.clean_data()
    
    top_skus = loader.get_top_skus(n=2)
    
    # SKU_A: 2 + 3 + 5 = 10 шт, SKU_B: 10 шт
    assert len(top_skus) == 2
    assert "SKU_A" in top_skus.index
    assert "SKU_B" in top_skus.index
    assert top_skus["SKU_A"] == 10