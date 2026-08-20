import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Добавляем корень проекта в sys.path для корректного импорта модуля src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


@pytest.fixture
def kaggle_sample_csv(tmp_path):
    """Фикстура для создания временного CSV-файла в формате Kaggle E-Commerce."""
    data = {
        "InvoiceNo": [
            "536365",   # Валидная транзакция (Неделя 1, 85123A)
            "536365",   # Валидная транзакция (Неделя 1, 85123A, другая цена)
            "536366",   # Валидная транзакция (Неделя 1, 71053)
            "536370",   # Валидная транзакция (Неделя 2, 85123A)
            "C536379",  # Возврат / отмена (начинается с 'C')
            "536380",   # Отрицательное количество (Quantity < 0)
            "536381",   # Нулевая цена (UnitPrice = 0.0)
            "536382",   # Служебный артикул POST
            "536383",   # Пропущенный артикул (NaN)
        ],
        "StockCode": [
            "85123A",
            "85123A",
            "71053",
            "85123A",
            "D",
            "85123A",
            "85123A",
            "POST",
            None,
        ],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "Discount",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "POSTAGE",
            "UNKNOWN ITEM",
        ],
        "Quantity": [6, 4, 10, 10, -1, -5, 2, 1, 3],
        "InvoiceDate": [
            "12/1/2026 8:26",
            "12/2/2026 10:00",
            "12/1/2026 8:30",
            "12/10/2026 14:00",
            "12/3/2026 11:00",
            "12/4/2026 12:00",
            "12/5/2026 13:00",
            "12/6/2026 15:00",
            "12/7/2026 16:00",
        ],
        "UnitPrice": [2.55, 3.00, 3.39, 2.50, 27.50, 2.55, 0.0, 18.0, 1.50],
        "CustomerID": [17850, 17850, 13047, 17850, 14000, 17850, 17850, 12415, 15000],
        "Country": ["United Kingdom"] * 9,
    }

    df = pd.DataFrame(data)
    csv_file = tmp_path / "kaggle_ecommerce_sample.csv"
    df.to_csv(csv_file, index=False, encoding="ISO-8859-1")
    return csv_file


# =====================================================================
# Тесты DataLoader
# =====================================================================

def test_load_data_success(kaggle_sample_csv):
    """Проверка успешной загрузки датасета и нормализации названий колонок."""
    loader = DataLoader(kaggle_sample_csv)
    df = loader.load_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9
    assert "Price" in df.columns  # UnitPrice переименовано в Price
    assert "InvoiceDate" in df.columns
    assert "StockCode" in df.columns
    assert "Quantity" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])


def test_load_data_file_not_found():
    """Проверка возбуждения FileNotFoundError при отсутствии файла."""
    loader = DataLoader("non_existent_path.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_load_data_missing_columns(tmp_path):
    """Проверка ошибки при отсутствии обязательных колонок."""
    bad_df = pd.DataFrame({"InvoiceNo": ["123"], "Description": ["Test"]})
    bad_file = tmp_path / "corrupted.csv"
    bad_df.to_csv(bad_file, index=False)

    loader = DataLoader(bad_file)
    with pytest.raises(KeyError, match="В датасете отсутствуют обязательные поля"):
        loader.load_data()


def test_clean_data(kaggle_sample_csv):
    """
    Проверка комплексной очистки:
    - фильтрация отмен (InvoiceNo 'C...'),
    - удаление служебных кодов ('POST', 'D'),
    - удаление неположительных цен и объемов,
    - заполнение словаря описаний.
    """
    loader = DataLoader(kaggle_sample_csv)
    loader.load_data()
    clean_df = loader.clean_data()

    # Из 9 исходных строк должны остаться только 4 валидные
    assert len(clean_df) == 4
    assert (clean_df["Quantity"] > 0).all()
    assert (clean_df["Price"] > 0).all()
    assert not clean_df["StockCode"].isin(["POST", "D"]).any()
    assert not clean_df["InvoiceNo"].str.startswith("C").any()

    # Проверка формирования справочника описаний
    assert "85123A" in loader.sku_descriptions
    assert loader.sku_descriptions["85123A"] == "WHITE HANGING HEART T-LIGHT HOLDER"


def test_get_aggregated_sku_data(kaggle_sample_csv):
    """
    Проверка корректности недельной агрегации и вычисления средневзвешенной цены.

    Для 85123A на первой неделе:
    - Покупка 1: 6 шт. по 2.55 (Сумма = 15.30)
    - Покупка 2: 4 шт. по 3.00 (Сумма = 12.00)
    - Всего: 10 шт.
    - Средневзвешенная цена = 27.30 / 10 = 2.73
    """
    loader = DataLoader(kaggle_sample_csv)
    loader.load_data()
    loader.clean_data()

    weekly_df = loader.get_aggregated_sku_data(sku="85123A", freq="W")

    assert len(weekly_df) == 2
    assert "Price" in weekly_df.columns
    assert "Quantity" in weekly_df.columns

    first_week = weekly_df.iloc[0]
    assert first_week["Quantity"] == 10
    assert np.isclose(first_week["Price"], 2.73, atol=1e-4)


def test_get_aggregated_sku_not_found(kaggle_sample_csv):
    """Проверка вызова исключения при запросе отсутствующего SKU."""
    loader = DataLoader(kaggle_sample_csv)
    loader.load_data()
    loader.clean_data()

    with pytest.raises(ValueError, match="не найден"):
        loader.get_aggregated_sku_data("UNKNOWN_SKU")


def test_get_top_skus(kaggle_sample_csv):
    """Проверка формирования топа товаров с человекочитаемыми описаниями."""
    loader = DataLoader(kaggle_sample_csv)
    loader.load_data()
    loader.clean_data()

    top_skus_df = loader.get_top_skus(n=2)

    assert isinstance(top_skus_df, pd.DataFrame)
    assert len(top_skus_df) == 2
    assert "StockCode" in top_skus_df.columns
    assert "Description" in top_skus_df.columns
    assert "TotalQuantity" in top_skus_df.columns
    assert "Label" in top_skus_df.columns

    # 85123A: 6 + 4 + 10 = 20 шт. (1 место)
    # 71053: 10 шт. (2 место)
    assert top_skus_df.iloc[0]["StockCode"] == "85123A"
    assert top_skus_df.iloc[0]["TotalQuantity"] == 20
    assert "WHITE HANGING HEART" in top_skus_df.iloc[0]["Label"]