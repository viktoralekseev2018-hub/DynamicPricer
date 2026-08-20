from pathlib import Path
from typing import Optional, Union, Dict, Any, cast
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки, очистки и агрегации транзакций датасета Kaggle E-Commerce.
    
    Оригинальная структура колонок:
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
    """

    # Служебные артикулы в датасете, не являющиеся реальными товарами
    SERVICE_CODES = {
        "POST", "D", "M", "PADS", "DOT", "CR", "BANK CHARGES", 
        "AMAZONFEE", "S", "B", "gift_0001_40", "gift_0001_50"
    }

    def __init__(self, file_path: Union[str, Path]):
        """
        :param file_path: Путь к CSV-файлу с транзакциями.
        """
        self.file_path = Path(file_path)
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.sku_descriptions: Dict[str, str] = {}

    def load_data(
        self,
        sep: str = ",",
        encoding: str = "ISO-8859-1"
    ) -> pd.DataFrame:
        """
        Загружает CSV-файл и нормализует форматы колонок.
        
        :param sep: Разделитель (по умолчанию запятая).
        :param encoding: Кодировка файла (по умолчанию ISO-8859-1 для Kaggle E-Commerce).
        :return: Загруженный DataFrame.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Файл не найден по адресу: {self.file_path}")

        logger.info(f"Загрузка данных из: {self.file_path} (кодировка: {encoding})")
        
        try:
            df = pd.read_csv(self.file_path, sep=sep, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            logger.warning("Сбой чтения с кодировкой ISO-8859-1, пробуем utf-8-sig...")
            df = pd.read_csv(self.file_path, sep=sep, encoding="utf-8-sig", low_memory=False)

        # Проверка обязательных колонок из Kaggle-датасета
        required_cols = ["InvoiceDate", "StockCode", "UnitPrice", "Quantity"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"В датасете отсутствуют обязательные поля: {missing}. Найдены колонки: {list(df.columns)}")

        # Переименовываем UnitPrice в стандартный Price для дальнейших модулей
        df = df.rename(columns={"UnitPrice": "Price"})

        # Приведение типов
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["StockCode"] = df["StockCode"].astype(str).str.strip()
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

        if "Description" in df.columns:
            df["Description"] = df["Description"].astype(str).str.strip()

        self.raw_df = df
        logger.info(f"Успешно загружено {len(df):,} строк.")
        return df

    def clean_data(self) -> pd.DataFrame:
        """
        Очищает данные:
        - Удаляет возвраты (Quantity <= 0) и отмены (InvoiceNo начинается с 'C').
        - Удаляет невалидные цены (Price <= 0).
        - Удаляет служебные артикулы (POST, MANUAL, D и т.д.).
        - Сохраняет словарь соответствия SKU -> Человекочитаемое название.
        """
        if self.raw_df is None:
            raise ValueError("Данные не загружены. Сначала вызовите .load_data().")

        df = self.raw_df.copy()
        initial_len = len(df)

        # 1. Удаление строк с пропущенными критическими значениями
        df = df.dropna(subset=["InvoiceDate", "StockCode", "Price", "Quantity"])

        # 2. Удаление отмененных чеков (InvoiceNo с префиксом 'C') и отрицательных объемов
        if "InvoiceNo" in df.columns:
            df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

        # 3. Фильтрация служебных кодов
        df = df[~df["StockCode"].str.upper().isin(self.SERVICE_CODES)]

        # 4. Формирование справочника названий товаров SKU -> Description
        if "Description" in df.columns:
            valid_desc = df[df["Description"].str.len() > 2]
            
            # Безопасное извлечение моды без риска IndexError
            raw_series = valid_desc.groupby("StockCode")["Description"].agg(
                lambda x: str(x.mode().iloc[0]) if not x.mode().empty else "No description"
            )
            # Явное приведение ключей и значений к str для удовлетворения Dict[str, str]
            self.sku_descriptions = {str(k): str(v) for k, v in raw_series.items()}

        self.clean_df = df
        removed = initial_len - len(df)
        logger.info(
            f"Очистка завершена. Удалено {removed:,} строк ({(removed / initial_len):.1%}). "
            f"Осталось: {len(df):,} строк."
        )
        return df

    def _get_clean_df(self) -> pd.DataFrame:
        """Гарантирует возврат не-None DataFrame для статического анализатора."""
        if self.clean_df is None:
            return self.clean_data()
        return self.clean_df

    def get_aggregated_sku_data(
        self,
        sku: str,
        freq: str = "W-SUN"
    ) -> pd.DataFrame:
        """
        Агрегирует продажи по конкретному товару по периодам (дням/неделям).
        Рассчитывает средневзвешенную цену: Sum(Price * Quantity) / Sum(Quantity).
        """
        clean_df = self._get_clean_df()

        sku_str = str(sku).strip()
        sku_df = clean_df[clean_df["StockCode"] == sku_str].copy()

        if sku_df.empty:
            raise ValueError(f"Товар с SKU '{sku_str}' не найден в данных.")

        # Вычисляем выручку по строке чека
        sku_df["Revenue"] = sku_df["Price"] * sku_df["Quantity"]

        # Агрегируем по окну времени
        aggregated = sku_df.groupby(pd.Grouper(key="InvoiceDate", freq=freq)).agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum")
        ).reset_index()

        # Оставляем периоды, где были продажи
        aggregated = aggregated[aggregated["Quantity"] > 0].copy()

        # Корректная средневзвешенная цена
        aggregated["Price"] = aggregated["Revenue"] / aggregated["Quantity"]
        aggregated = aggregated.drop(columns=["Revenue"]).reset_index(drop=True)

        logger.info(f"Сформирован ряд для SKU '{sku_str}': {len(aggregated)} точек.")
        return aggregated

    def get_top_skus(self, n: int = 10) -> pd.DataFrame:
        """
        Возвращает топ-N товаров по общему объему продаж вместе с описанием.
        """
        clean_df = self._get_clean_df()

        top_series = clean_df.groupby("StockCode")["Quantity"].sum().nlargest(n)
        
        result = []
        for sku_key, total_q in top_series.items():
            sku_str = str(sku_key)
            desc = self.sku_descriptions.get(sku_str, "Без описания")
            result.append({
                "StockCode": sku_str,
                "Description": desc,
                "TotalQuantity": int(cast(Any, total_q)),
                "Label": f"{sku_str} — {desc}"
            })
            
        return pd.DataFrame(result)


if __name__ == "__main__":
    dataset_file = "data/data.csv"
    
    try:
        loader = DataLoader(dataset_file)
        loader.load_data()
        loader.clean_data()
        
        print("\n--- Топ-5 продаваемых товаров из Kaggle ---")
        top_df = loader.get_top_skus(5)
        for _, row in top_df.iterrows():
            print(f"[{row['StockCode']}] {row['Description']} — {int(row['TotalQuantity']):,} шт.")
            
        first_sku = str(top_df.iloc[0]["StockCode"])
        sku_series = loader.get_aggregated_sku_data(sku=first_sku, freq="W-SUN")
        print(f"\nАгрегированные продажи для {first_sku}:")
        print(sku_series.head())
        
    except Exception as e:
        print(f"Ошибка проверки: {e}")