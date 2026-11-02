import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import (
    load_transactions,
    load_customer_features,
    positive_sales,
)
from src.analytics import (
    sales_summary,
    monthly_sales,
    top_products,
    top_countries,
)
from src.segmentation import segment_customers
from src.churn import add_proxy_risk
from src.forecasting import build_forecast
from src.inventory import inventory_recommendations


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetailPulse | Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# APPLICATION THEME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background and default text */
    .stApp {
        background-color: #F4F7FC;
        color: #263B59;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #17365D !important;
        font-weight: 700 !important;
    }

    p, li, label {
        color: #263B59;
    }

    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #302783 0%,
            #4335B5 55%,
            #6254E8 100%
        );
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.25);
    }

    /* Radio navigation */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #FFFFFF !important;
        padding: 5px 8px;
        border-radius: 8px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5F1;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 3px 12px rgba(25, 45, 80, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #526783 !important;
    }

    [data-testid="stMetricValue"] {
        color: #17365D !important;
        font-weight: 750;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #DCE5F1;
        border-radius: 10px;
        background: #FFFFFF;
    }

    /* Buttons */
    .stDownloadButton button,
    .stButton button {
        background: #4338CA;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    .stDownloadButton button:hover,
    .stButton button:hover {
        background: #312E81;
        color: #FFFFFF !important;
        border: none;
    }

    /* Inputs */
    .stSelectbox, .stSlider, .stRadio {
        color: #263B59;
    }

    /* Keep Streamlit header transparent */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Caption contrast */
    [data-testid="stCaptionContainer"] {
        color: #526783 !important;
    }

    /* Links */
    a {
        color: #4338CA !important;
    }

    /* Sidebar collapse/expand buttons — visible in both states */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] {
        background-color: #6C63FF !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stExpandSidebarButton"] button {
        background-color: #6C63FF !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stExpandSidebarButton"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stExpandSidebarButton"] button:hover {
        background-color: #5145CD !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CHART STYLE
# ---------------------------------------------------------
PURPLE = "#6254E8"
DARK_PURPLE = "#4338CA"
TEAL = "#009F91"
BLUE = "#2684FF"
ORANGE = "#F59E0B"
RED = "#E05260"

SEGMENT_COLORS = {
    "Lapsed": "#E05260",
    "At Risk": "#F59E0B",
    "Engaged": "#2684FF",
    "Recent/Active": "#009F91",
    "Lapsed/At Risk": "#E05260",
    "Lower Engagement": "#F59E0B",
    "Higher Engagement": "#009F91",
}


def style_chart(fig, height=420):
    """Apply readable labels and a consistent white chart background."""
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#263B59",
        ),
        title=dict(
            font=dict(size=18, color="#17365D"),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(
            title_font=dict(size=13, color="#263B59"),
            tickfont=dict(size=12, color="#263B59"),
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1",
        ),
        yaxis=dict(
            title_font=dict(size=13, color="#263B59"),
            tickfont=dict(size=12, color="#263B59"),
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1",
        ),
        legend=dict(
            font=dict(size=12, color="#263B59"),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=25, r=25, t=65, b=45),
        hoverlabel=dict(
            bgcolor="#17365D",
            font=dict(color="#FFFFFF", size=13),
        ),
    )
    return fig


def show_chart(fig, height=420):
    """Display a chart using the shared styling."""
    st.plotly_chart(
        style_chart(fig, height),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )


def section_heading(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def money(value, decimals=0):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 18px 0;">
            <h2 style="color:white !important; margin-bottom:4px;">
                🛒 RetailPulse
            </h2>
            <p style="
                color:#E2E8FF !important;
                font-size:12px;
                letter-spacing:1.4px;
                font-weight:600;
                margin-top:0;
            ">
                RETAIL INTELLIGENCE PLATFORM
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Navigation")

    pages = [
        "🏠 Executive Dashboard",
        "📈 Sales Analytics",
        "👥 Customer Analytics",
        "🎯 Customer Segmentation",
        "⚠️ Churn Risk",
        "🔮 Demand Forecast",
        "📦 Inventory",
    ]

    selected_page = st.radio(
        "Choose a page",
        pages,
        label_visibility="collapsed",
    )

    page = selected_page.split(" ", 1)[1]

    st.divider()
    st.markdown("### DATA SOURCE")
    st.caption("UCI Online Retail")
    st.caption("Currency: original dataset currency")
    st.divider()
    st.caption("RetailPulse · Local demonstration")


# ---------------------------------------------------------
# LOAD DATA WITH CACHING
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading transaction data...")
def get_transactions():
    return load_transactions()


@st.cache_data(show_spinner="Loading customer features...")
def get_customers():
    return load_customer_features()


try:
    df = get_transactions()
    customers = get_customers()

    if df.empty:
        st.error("The transaction dataset is empty.")
        st.stop()

    if customers.empty:
        st.error("The customer-feature dataset is empty.")
        st.stop()

except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.info(
        "Check that these files exist:\n"
        "- data/processed/cleaned_retail.csv\n"
        "- data/processed/customer_features.csv"
    )
    st.stop()


# Calculate only what the selected page needs where possible.
summary = None


# ---------------------------------------------------------
# 1. EXECUTIVE DASHBOARD
# ---------------------------------------------------------
if page == "Executive Dashboard":

    st.markdown(
        """
        <p style="
            color:#009F91;
            font-size:12px;
            font-weight:700;
            letter-spacing:2px;
            margin-bottom:0;
        ">RETAILPULSE ANALYTICS</p>
        """,
        unsafe_allow_html=True,
    )

    st.title("Executive Dashboard")
    st.write("Monitor sales performance, products, and customer reach.")

    with st.spinner("Preparing executive overview..."):
        summary = sales_summary(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Positive-sales value",
        money(summary["positive_sales_value"]),
    )
    c2.metric(
        "Units sold",
        f'{summary["positive_units"]:,}',
    )
    c3.metric(
        "Purchase invoices",
        f'{summary["invoices"]:,}',
    )
    c4.metric(
        "Identifiable customers",
        f'{summary["customers"]:,}',
    )

    st.markdown("### 📈 Sales performance over time")

    monthly = monthly_sales(df)

    if not monthly.empty:
        trend_fig = px.line(
            monthly,
            x="Month",
            y="SalesValue",
            markers=True,
            title="Monthly positive-sales value",
            color_discrete_sequence=[TEAL],
        )
        trend_fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
        )
        show_chart(trend_fig, 430)
    else:
        st.info("No monthly sales data is available.")

    left, right = st.columns(2)

    with left:
        product_data = top_products(df, 10)
        product_fig = px.bar(
            product_data,
            x="SalesValue",
            y="Description",
            orientation="h",
            title="🏆 Top products by sales",
            color_discrete_sequence=[PURPLE],
        )
        product_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_chart(product_fig, 450)

    with right:
        country_data = top_countries(df, 10)
        country_fig = px.bar(
            country_data,
            x="SalesValue",
            y="Country",
            orientation="h",
            title="🌍 Top countries by sales",
            color_discrete_sequence=[TEAL],
        )
        country_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_chart(country_fig, 450)

    st.caption(
        "Positive sales exclude non-positive quantities and cancellation invoices. "
        "This is an analytical measure, not audited accounting revenue."
    )


# ---------------------------------------------------------
# 2. SALES ANALYTICS
# ---------------------------------------------------------
elif page == "Sales Analytics":

    st.title("📈 Sales Analytics")
    st.write("Explore monthly sales trends, leading products, and country performance.")

    with st.spinner("Preparing sales analytics..."):
        monthly = monthly_sales(df)
        product_data = top_products(df, 15)
        country_data = top_countries(df, 15)

    if not monthly.empty:
        trend_fig = px.line(
            monthly,
            x="Month",
            y="SalesValue",
            markers=True,
            title="Monthly sales trend",
            color_discrete_sequence=[TEAL],
        )
        trend_fig.update_traces(line=dict(width=3))
        show_chart(trend_fig, 420)

    col1, col2 = st.columns(2)

    with col1:
        product_fig = px.bar(
            product_data,
            x="SalesValue",
            y="Description",
            orientation="h",
            title="Top 15 products",
            color_discrete_sequence=[PURPLE],
        )
        product_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_chart(product_fig, 520)

    with col2:
        country_fig = px.bar(
            country_data,
            x="SalesValue",
            y="Country",
            orientation="h",
            title="Top 15 countries",
            color_discrete_sequence=[TEAL],
        )
        country_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_chart(country_fig, 520)

    st.markdown("### Monthly sales data")
    st.dataframe(
        monthly.round(2),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# 3. CUSTOMER ANALYTICS
# ---------------------------------------------------------
elif page == "Customer Analytics":

    st.title("👥 Customer Analytics")
    st.write("Understand customer purchasing behaviour and value.")

    required = {"RecencyDays", "Frequency", "Monetary"}
    missing = required - set(customers.columns)

    if missing:
        st.error(f"Customer data is missing required columns: {sorted(missing)}")
        st.stop()

    recency = pd.to_numeric(customers["RecencyDays"], errors="coerce")
    frequency = pd.to_numeric(customers["Frequency"], errors="coerce")
    monetary = pd.to_numeric(customers["Monetary"], errors="coerce")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(customers):,}")
    c2.metric("Median recency", f"{recency.median():,.1f} days")
    c3.metric("Median monetary value", money(monetary.median(), 2))

    left, right = st.columns(2)

    with left:
        recency_fig = px.histogram(
            customers.assign(RecencyDays=recency),
            x="RecencyDays",
            nbins=40,
            title="Customer recency distribution",
            color_discrete_sequence=[PURPLE],
        )
        show_chart(recency_fig, 420)

    with right:
        scatter_fig = px.scatter(
            customers.assign(Frequency=frequency, Monetary=monetary),
            x="Frequency",
            y="Monetary",
            color="RecencyDays",
            hover_data=["CustomerID"] if "CustomerID" in customers else None,
            title="Frequency vs monetary value",
            color_continuous_scale="Viridis",
        )
        show_chart(scatter_fig, 420)

    st.markdown("### Highest-value customers")
    customer_table = customers.copy()
    customer_table["Monetary"] = monetary
    st.dataframe(
        customer_table.sort_values("Monetary", ascending=False).head(30),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# 4. CUSTOMER SEGMENTATION
# ---------------------------------------------------------
elif page == "Customer Segmentation":

    st.title("🎯 Customer Segmentation")
    st.write("Explore RFM segments and customer clusters.")

    required = {"RecencyDays", "Frequency", "Monetary"}
    missing = required - set(customers.columns)

    if missing:
        st.error(f"Customer data is missing required columns: {sorted(missing)}")
        st.stop()

    k = st.slider(
        "Number of customer clusters",
        min_value=2,
        max_value=6,
        value=4,
        key="segment_count",
    )

    try:
        with st.spinner("Calculating customer segments..."):
            segmented = segment_customers(customers, k)

        valid_segments = segmented.dropna(subset=["SegmentName"]).copy()

        if valid_segments.empty:
            st.warning("No valid customers were available for segmentation.")
            st.stop()

        avg_recency = valid_segments["RecencyDays"].mean()
        avg_frequency = valid_segments["Frequency"].mean()
        avg_monetary = valid_segments["Monetary"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{len(valid_segments):,}")
        c2.metric("Average recency", f"{avg_recency:,.1f} days")
        c3.metric("Average frequency", f"{avg_frequency:,.2f}")
        c4.metric("Average monetary value", money(avg_monetary, 2))
        sil_score = segmented.attrs.get("silhouette_score")

        if sil_score is not None:
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        else:
            st.info("Silhouette Score is unavailable for this clustering result.")

        left, right = st.columns(2)

        with left:
            segment_counts = (
                valid_segments["SegmentName"]
                .value_counts()
                .rename_axis("Segment")
                .reset_index(name="Customers")
            )

            donut_fig = px.pie(
                segment_counts,
                names="Segment",
                values="Customers",
                hole=0.58,
                title="Customer segment distribution",
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
            )
            donut_fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                insidetextfont=dict(color="#FFFFFF", size=11),
            )
            show_chart(donut_fig, 440)

        with right:
            cluster_fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                color="Segment",
                title="Customer cluster distribution",
                color_discrete_map=SEGMENT_COLORS,
                text="Customers",
            )
            cluster_fig.update_traces(textposition="outside")
            show_chart(cluster_fig, 440)

        st.markdown("### RFM customer profile")

        profile = (
            valid_segments.groupby("SegmentName", dropna=False)
            .agg(
                Customers=("SegmentName", "size"),
                AvgRecency=("RecencyDays", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean"),
            )
            .reset_index()
            .round(2)
        )

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Customer cluster scatter plot")

        scatter_fig = px.scatter(
            valid_segments,
            x="Frequency",
            y="Monetary",
            color="SegmentName",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=[
                c for c in ["CustomerID", "RecencyDays", "Frequency", "Monetary"]
                if c in valid_segments.columns
            ],
            title="Customer clusters by frequency and monetary value",
            log_x=True,
            log_y=True,
        )
        show_chart(scatter_fig, 520)

        st.download_button(
            "⬇️ Download segmented customers CSV",
            data=segmented.to_csv(index=False).encode("utf-8"),
            file_name="customer_segments.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Segmentation failed: {exc}")


# ---------------------------------------------------------
# 5. CHURN RISK
# ---------------------------------------------------------
elif page == "Churn Risk":

    st.title("⚠️ Customer Churn Risk")
    st.write("Identify customers who may need follow-up based on purchase recency.")

    if "RecencyDays" not in customers.columns:
        st.error("Customer data must contain RecencyDays.")
        st.stop()

    threshold = st.slider(
        "At-risk recency threshold (days)",
        min_value=30,
        max_value=180,
        value=90,
        step=15,
        key="churn_threshold",
    )

    risk = add_proxy_risk(customers, threshold)

    st.warning(
        "This is a recency-based risk proxy, not a validated churn prediction. "
        "The source dataset does not contain confirmed churn labels."
    )

    counts = (
        risk["ChurnRisk"]
        .value_counts()
        .rename_axis("Risk")
        .reset_index(name="Customers")
    )

    risk_colors = {
        "Low": TEAL,
        "Medium": ORANGE,
        "High": RED,
        "nan": "#94A3B8",
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers analysed", f"{len(risk):,}")
    c2.metric("High-risk customers", f"{(risk['ChurnRisk'] == 'High').sum():,}")
    c3.metric("Threshold", f"{threshold} days")

    left, right = st.columns(2)

    with left:
        risk_fig = px.bar(
            counts,
            x="Risk",
            y="Customers",
            color="Risk",
            color_discrete_map=risk_colors,
            title="Customers by proxy risk",
            text="Customers",
        )
        risk_fig.update_traces(textposition="outside")
        show_chart(risk_fig, 420)

    with right:
        if "Monetary" in risk.columns:
            risk_fig = px.scatter(
                risk,
                x="RecencyDays",
                y="Monetary",
                color="ChurnRisk",
                color_discrete_map=risk_colors,
                hover_data=["CustomerID"] if "CustomerID" in risk else None,
                title="Customer recency vs monetary value",
            )
            show_chart(risk_fig, 420)

    st.markdown("### Customers requiring review")

    st.dataframe(
        risk.sort_values("RecencyDays", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download churn-risk CSV",
        data=risk.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_risk.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# 6. DEMAND FORECAST
# ---------------------------------------------------------
elif page == "Demand Forecast":

    st.title("🔮 Demand Forecast")
    st.write("Estimate future monthly sales using the available forecasting model.")

    periods = st.slider(
        "Forecast horizon (months)",
        min_value=1,
        max_value=12,
        value=3,
        key="forecast_periods",
    )

    try:
        with st.spinner("Generating forecast..."):
            forecast, model_name = build_forecast(df, periods)

        st.info(f"Forecast engine: {model_name}")

        if "Type" in forecast.columns:
            actual = forecast[forecast["Type"] == "Actual"]
            future = forecast[forecast["Type"] == "Forecast"]
        else:
            actual = forecast.iloc[0:0]
            future = forecast

        c1, c2 = st.columns(2)
        c1.metric("Forecast months", f"{periods}")
        c2.metric(
            "Average forecast sales",
            money(pd.to_numeric(future["yhat"], errors="coerce").mean(), 2)
            if not future.empty and "yhat" in future.columns
            else "N/A",
        )

        fig = px.line(
            forecast,
            x="ds",
            y="yhat",
            title="Monthly sales forecast",
            color_discrete_sequence=[PURPLE],
        )
        fig.update_traces(line=dict(width=3), name="Forecast / fitted trend")

        if not actual.empty and "y" in actual.columns:
            fig.add_scatter(
                x=actual["ds"],
                y=actual["y"],
                mode="markers",
                name="Actual sales",
                marker=dict(color=TEAL, size=7),
            )

        if (
            "yhat_lower" in forecast.columns
            and "yhat_upper" in forecast.columns
            and forecast["yhat_lower"].notna().any()
        ):
            fig.add_scatter(
                x=forecast["ds"],
                y=forecast["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Upper interval",
            )
            fig.add_scatter(
                x=forecast["ds"],
                y=forecast["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(98,84,232,0.15)",
                name="Forecast interval",
            )

        show_chart(fig, 480)

        st.markdown("### Forecast results")

        forecast_columns = [
            c for c in ["ds", "yhat", "yhat_lower", "yhat_upper"]
            if c in forecast.columns
        ]

        st.dataframe(
            forecast.tail(periods)[forecast_columns].round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download forecast CSV",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name="demand_forecast.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Forecast could not be generated: {exc}")


# ---------------------------------------------------------
# 7. INVENTORY
# ---------------------------------------------------------
elif page == "Inventory":

    st.title("📦 Inventory Optimization")
    st.write("Review product demand and estimate replenishment requirements.")

    col1, col2 = st.columns(2)

    with col1:
        lead = st.slider(
            "Assumed supplier lead time (days)",
            min_value=1,
            max_value=60,
            value=14,
            key="inventory_lead",
        )

    with col2:
        safety = st.slider(
            "Safety-stock buffer (days)",
            min_value=1,
            max_value=30,
            value=7,
            key="inventory_safety",
        )

    st.warning(
        "These are demand-based planning estimates. The source dataset does not "
        "contain current stock-on-hand or verified supplier lead-time information."
    )

    try:
        with st.spinner("Calculating inventory recommendations..."):
            inventory = inventory_recommendations(
                df,
                lead_time_days=lead,
                safety_days=safety,
                top_n=50,
            )

        if inventory.empty:
            st.info("No inventory recommendations are available.")
        else:
            c1, c2, c3 = st.columns(3)

            c1.metric("Products analysed", f"{len(inventory):,}")
            c2.metric(
                "Units sold in displayed products",
                f'{inventory["UnitsSold"].sum():,.0f}',
            )
            c3.metric(
                "Products prioritised",
                f'{(inventory["SuggestedAction"] == "Prioritize replenishment").sum():,}',
            )

            inventory_chart = inventory.head(20).sort_values("UnitsSold")

            fig = px.bar(
                inventory_chart,
                x="UnitsSold",
                y="Description",
                orientation="h",
                title="Top products by units sold",
                color_discrete_sequence=[TEAL],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            show_chart(fig, 520)

            st.markdown("### Inventory recommendations")

            st.dataframe(
                inventory.round(2),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download inventory recommendations",
                data=inventory.to_csv(index=False).encode("utf-8"),
                file_name="inventory_recommendations.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Inventory recommendations failed: {exc}")


# ---------------------------------------------------------
# ABOUT SECTION
# ---------------------------------------------------------
with st.expander("ℹ️ About this prototype"):
    st.write(
        "RetailPulse is a local retail analytics prototype built using the "
        "provided cleaned transaction and customer-feature datasets. It includes "
        "sales analytics, RFM customer segmentation, recency-based churn-risk "
        "indicators, demand forecasting, and demand-based inventory estimates. "
        "Validate model assumptions and forecast performance before using results "
        "for real business decisions."
    )

st.markdown("""
<style>
/* Change only the sidebar radio selection indicator */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
}

/* Selected navigation indicator */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div:first-child {
    background-color: #ff4b4b !important;
    border-color: #ff4b4b !important;
}
</style>
""", unsafe_allow_html=True)