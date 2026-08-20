import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Добавляем корневую папку в sys.path для корректных импортов
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.demand_model import DemandModel
from src.optimizer import ProfitOptimizer


# =====================================================================
# 1. Настройка страницы
# =====================================================================
st.set_page_config(
    page_title="DynamicPricer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 2. Вспомогательная функция: генерация демо-данных
# =====================================================================
@st.cache_data
def generate_mock_data() -> pd.DataFrame:
    """Генерирует реалистичный датасет для демонстрации."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=52, freq="W-SUN")
    skus = ["SKU_85123A (Винтажный светильник)", "SKU_22423 (Чайный сервиз)", "SKU_47566 (Плед шерстой)"]
    
    records = []
    configs = {
        skus[0]: (1200.0, -1.85),
        skus[1]: (850.0, -0.75),
        skus[2]: (2400.0, -2.20)
    }

    for sku, (base_p, beta) in configs.items():
        prices = base_p * np.random.uniform(0.85, 1.20, len(dates))
        alpha = 14.0 if beta < -1 else 7.0
        log_q = alpha + beta * np.log(prices) + np.random.normal(0, 0.12, len(dates))
        quantities = np.maximum(np.exp(log_q).astype(int), 1)

        for d, p, q in zip(dates, prices, quantities):
            records.append({
                "InvoiceDate": d,
                "StockCode": sku,
                "Price": round(float(p), 2),
                "Quantity": int(q)
            })

    return pd.DataFrame(records)


# =====================================================================
# 3. Боковая панель (Sidebar): Загрузка и настройки
# =====================================================================
st.sidebar.title("⚙️ Управление проектом")

data_source = st.sidebar.radio(
    "Источник данных:",
    ("Демо-данные (быстрый старт)", "Загрузить свой CSV")
)

df_clean: Optional[pd.DataFrame] = None

if data_source == "Загрузить свой CSV":
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV-файл транзакций", type=["csv"])
    if uploaded_file is not None:
        try:
            temp_path = Path("temp_uploaded.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = DataLoader(temp_path)
            loader.load_data()
            df_clean = loader.clean_data()
            st.sidebar.success(f"Загружено {len(df_clean):,} валидных строк!")
        except Exception as e:
            st.sidebar.error(f"Ошибка чтения файла: {e}")
    else:
        st.info("Пожалуйста, загрузите CSV файл или переключитесь на «Демо-данные».")
        st.stop()
else:
    df_clean = generate_mock_data()

# Защита от None для Pylance
if df_clean is None:
    st.stop()

# Выбор SKU
available_skus = sorted(df_clean["StockCode"].astype(str).unique())
selected_sku = st.sidebar.selectbox("Выберите товар (SKU):", available_skus)

# Фильтрация данных по SKU
sku_df = df_clean[df_clean["StockCode"] == selected_sku].copy()
sku_df = sku_df.groupby("InvoiceDate").agg(
    Price=("Price", "mean"),
    Quantity=("Quantity", "sum")
).reset_index()

current_avg_price = float(sku_df["Price"].mean())

st.sidebar.markdown("---")
st.sidebar.subheader("Параметры оптимизации")
unit_cost = st.sidebar.number_input(
    "Себестоимость единицы (руб.):",
    min_value=1.0,
    max_value=max(current_avg_price * 2, 100.0),
    value=float(round(current_avg_price * 0.55, 2)),
    step=10.0
)

min_margin = st.sidebar.slider("Мин. целевая маржинальность (%):", 5, 50, 15) / 100.0
max_change = st.sidebar.slider("Лимит изменения цены (+/- %):", 5, 50, 20) / 100.0


# =====================================================================
# 4. Основной экран: Обучение модели и Оптимизация
# =====================================================================
st.title("📈 DynamicPricer: Анализ эластичности и оптимизация цен")
st.caption("Система поддержки принятия решений на основе регрессионного моделирования спроса")

try:
    # 1. Обучаем модель спроса
    model = DemandModel(min_samples=5)
    model.fit(sku_df, price_col="Price", quantity_col="Quantity")

    if model.beta is None or model.alpha is None:
        raise ValueError("Параметры модели не были рассчитаны.")

    # 2. Запускаем оптимизатор прибыли
    optimizer = ProfitOptimizer(demand_model=model)
    opt_res = optimizer.optimize_price(
        unit_cost=float(unit_cost),
        current_price=current_avg_price,
        min_margin=min_margin,
        max_price_change=max_change
    )
except Exception as e:
    st.error(f"Не удалось обучить модель для выбранного товара: {e}")
    st.stop()

# =====================================================================
# 5. Верхний блок: Метрики и KPI
# =====================================================================
st.markdown("### 📊 Ключевые показатели эффективности (KPI)")

col1, col2, col3, col4 = st.columns(4)

delta_price_pct = ((opt_res["optimal_price"] / opt_res["current_price"]) - 1) * 100
col1.metric(
    label="Текущая / Рекомендованная цена",
    value=f"{opt_res['optimal_price']:.2f} ₽",
    delta=f"{delta_price_pct:+.1f}% от {opt_res['current_price']:.2f} ₽",
    delta_color="normal"
)

col2.metric(
    label="Прогнозируемый прирост прибыли",
    value=f"{opt_res['expected_profit']:,.0f} ₽",
    delta=f"+{opt_res['profit_uplift_pct']:.1%} Uplift",
    delta_color="normal"
)

col3.metric(
    label="Ценовая эластичность (β)",
    value=f"{model.beta:.2f}",
    delta="Эластичный" if abs(model.beta) > 1 else "Неэластичный",
    delta_color="off"
)

col4.metric(
    label="Качество модели (R²)",
    value=f"{model.metrics['r2']:.2f}",
    delta=f"MAPE: {model.metrics['mape']:.1%}",
    delta_color="off"
)

st.info(f"💡 **Вывод аналитики:** {model.get_elasticity_interpretation()}")

# =====================================================================
# 6. Интерактивные графики Plotly
# =====================================================================
tab1, tab2 = st.tabs(["🎯 Кривые спроса и Sweet Spot", "🕒 Динамика продаж во времени"])

with tab1:
    p_min_plot = max(unit_cost * 0.9, current_avg_price * 0.6)
    p_max_plot = current_avg_price * 1.5
    price_grid = np.linspace(p_min_plot, p_max_plot, 150)
    
    demand_curve = np.asarray(model.predict(price_grid), dtype=float)
    profit_curve = (price_grid - unit_cost) * demand_curve

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price_grid, y=demand_curve,
        mode="lines", name="Спрос Q(P) [шт.]",
        line=dict(color="#1f77b4", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=sku_df["Price"], y=sku_df["Quantity"],
        mode="markers", name="История продаж",
        marker=dict(color="#1f77b4", size=7, opacity=0.6)
    ))

    fig.add_trace(go.Scatter(
        x=price_grid, y=profit_curve,
        mode="lines", name="Прибыль Profit(P) [₽]",
        line=dict(color="#2ca02c", width=3, dash="dash"),
        yaxis="y2"
    ))

    fig.add_vline(
        x=opt_res["optimal_price"],
        line_width=2, line_dash="dot", line_color="red",
        annotation_text=f"Sweet Spot: {opt_res['optimal_price']:.1f} ₽",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Оптимизация: Соотношение цены, спроса и маржинальности",
        xaxis=dict(title="Розничная цена (₽)"),
        yaxis=dict(title="Объем спроса (шт.)", title_font=dict(color="#1f77b4")),
        yaxis2=dict(
            title="Прибыль (₽)",
            title_font=dict(color="#2ca02c"),
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=sku_df["InvoiceDate"], y=sku_df["Price"],
        name="Цена (₽)", mode="lines+markers", line=dict(color="#ff7f0e")
    ))
    fig_time.add_trace(go.Scatter(
        x=sku_df["InvoiceDate"], y=sku_df["Quantity"],
        name="Продажи (шт.)", mode="lines+markers", yaxis="y2", line=dict(color="#1f77b4")
    ))
    fig_time.update_layout(
        title="Исторический временной ряд",
        xaxis=dict(title="Дата"),
        yaxis=dict(title="Цена (₽)", title_font=dict(color="#ff7f0e")),
        yaxis2=dict(title="Объем продаж (шт.)", title_font=dict(color="#1f77b4"), overlaying="y", side="right"),
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_time, use_container_width=True)

# =====================================================================
# 7. Модуль интерактивного симулятора («What-If Analysis»)
# =====================================================================
st.markdown("---")
st.subheader("🧪 Симулятор сценариев («What-If Analysis»)")
st.write("Проверьте, как изменится финансовый результат, если вы вручную установите свою цену:")

sim_price = st.slider(
    "Задайте тестовую цену товара (₽):",
    min_value=float(round(p_min_plot, 1)),
    max_value=float(round(p_max_plot, 1)),
    value=float(round(opt_res["optimal_price"], 1)),
    step=5.0
)

sim_q = float(np.asarray(model.predict(sim_price)).item())
sim_profit = (sim_price - unit_cost) * sim_q
sim_uplift = (sim_profit - opt_res["current_profit"]) / opt_res["current_profit"] if opt_res["current_profit"] > 0 else 0.0

scol1, scol2, scol3 = st.columns(3)
scol1.metric("Ожидаемые продажи", f"{int(sim_q)} шт.", delta=f"{int(sim_q - opt_res['current_quantity'])} шт.")
scol2.metric("Ожидаемая выручка", f"{(sim_price * sim_q):,.0f} ₽")
scol3.metric("Чистая маржа", f"{sim_profit:,.0f} ₽", delta=f"{sim_uplift:+.1%} vs текущая цена")