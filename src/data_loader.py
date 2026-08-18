from pathlib import Path
from typing import Optional, Union
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки, очистки и агрегации транзакционных данных E-Commerce / Retail.
    
    Поддерживает работу с датасетами формата UCI Online Retail, Kaggle Superstore 
    и другими стандартными CSV-выгрузками транзакций.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Инициализация загрузчика данных.
        
        :param file_path: Путь к CSV-файлу с транзакциями.
        """
        self.file_path = Path(file_path)
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

    def load_data(
        self,
        sep: str = ",",
        encoding: str = "utf-8",
        date_col: str = "InvoiceDate",
        sku_col: str = "StockCode",
        price_col: str = "Price",
        quantity_col: str = "Quantity"
    ) -> pd.DataFrame:
        """
        Загружает CSV-файл и приводит основные колонки к единому формату.
        
        :param sep: Разделитель колонок в CSV.
        :param encoding: Кодировка файла.
        :param date_col: Название колонки с датой.
        :param sku_col: Название колонки с артикулом/идентификатором товара (SKU).
        :param price_col: Название колонки с ценой.
        :param quantity_col: Название колонки с количеством.
        :return: Загруженный DataFrame.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Файл не найден по адресу: {self.file_path}")

        logger.info(f"Чтение файла: {self.file_path}")
        df = pd.read_csv(self.file_path, sep=sep, encoding=encoding)

        # Переименование колонок к внутреннему стандарту
        rename_map = {
            date_col: "InvoiceDate",
            sku_col: "StockCode",
            price_col: "Price",
            quantity_col: "Quantity"
        }
        df = df.rename(columns=rename_map)

        required_cols = ["InvoiceDate", "StockCode", "Price", "Quantity"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"В датасете отсутствуют обязательные колонки: {missing_cols}")

        # Приведение типов
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["StockCode"] = df["StockCode"].astype(str).str.strip()
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

        self.raw_df = df
        logger.info(f"Успешно загружено {len(df)} строк.")
        return df

    def clean_data(self) -> pd.DataFrame:
        """
        Очищает датасет от аномалий, пропусков, возвратов и невалидных цен.
        
        :return: Очищенный DataFrame.
        """
        if self.raw_df is None:
            raise ValueError("Данные еще не загружены. Сначала вызовите метод load_data().")
        df = self.raw_df.copy()
        initial_len = len(df)
        df = df.dropna(subset=["InvoiceDate", "StockCode", "Price", "Quantity"])
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
        q_low = df["Price"].quantile(0.001)
        q_high = df["Price"].quantile(0.999)
        df = df[(df["Price"] >= q_low) & (df["Price"] <= q_high)]
        self.clean_df = df
        removed_cnt = initial_len - len(df)
        logger.info(f"Очистка завершена. Удалено {removed_cnt} невалидных записей ({(removed_cnt / initial_len):.2%}). Осталось: {len(df)}.")
        return df

    def get_aggregated_sku_data(
        self,
        sku: str,
        freq: str = "W"
    ) -> pd.DataFrame:
        """
        Фильтрует данные по конкретной SKU и агрегирует по временным интервалам.
        """
        if self.clean_df is None:
            logger.warning("Clean data отсутствует, автоматически запускаем clean_data().")
        
        df = self.clean_df if self.clean_df is not None else self.clean_data()

        sku_str = str(sku).strip()
        sku_df = df[df["StockCode"] == sku_str].copy()

        if sku_df.empty:
            raise ValueError(f"Товар с SKU '{sku_str}' не найден в очищенных данных.")

        # Расчет выручки по каждой транзакции для вычисления средневзвешенной цены
        sku_df["Revenue"] = sku_df["Price"] * sku_df["Quantity"]

        # Агрегация по временному окну
        aggregated = sku_df.groupby(pd.Grouper(key="InvoiceDate", freq=freq)).agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            Price=("Price", "mean")
        ).reset_index()

        # Корректный расчет средневзвешенной цены за период (Revenue / Quantity)
        valid_sales = aggregated["Quantity"] > 0
        aggregated.loc[valid_sales, "Price"] = (
            aggregated.loc[valid_sales, "Revenue"] / aggregated.loc[valid_sales, "Quantity"]
        )

        # Удаление периодов без продаж
        aggregated = aggregated[aggregated["Quantity"] > 0].drop(columns=["Revenue"]).reset_index(drop=True)

        logger.info(f"Сформирован временной ряд для SKU '{sku_str}': {len(aggregated)} точек (freq='{freq}').")
        return aggregated

    def get_top_skus(self, n: int = 10) -> pd.Series:
        """
        Возвращает топ-N популярных товаров по объему продаж.
        """
        df = self.clean_df if self.clean_df is not None else self.clean_data()
        return df.groupby("StockCode")["Quantity"].sum().nlargest(n)


if __name__ == "__main__":
    # Пример использования (для быстрого локального ручного тестирования)
    import sys

    sample_csv = "data/online_retail_II.csv"
    
    try:
        loader = DataLoader(sample_csv)
        loader.load_data()
        loader.clean_data()
        
        top_sku = loader.get_top_skus(1).index[0]
        print(f"Самый продаваемый SKU: {top_sku}")
        
        sku_series = loader.get_aggregated_sku_data(sku=top_sku, freq="W")
        print("\nПервые 5 строк агрегированного ряда:")
        print(sku_series.head())
    except Exception as e:
        print(f"[Демо-режим] Для тестирования положительный CSV по пути '{sample_csv}'. Ошибка: {e}")