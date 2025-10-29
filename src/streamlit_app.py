from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import polars as pl
import streamlit as st
import plotly.express as px
from polars.exceptions import PolarsError

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_METRICS: list[str] = [
    "retweetCount",
    "replyCount",
    "likeCount",
    "quoteCount",
    "bookmarkCount",
    "viewCount",
]
AUTHOR_NUMERIC: list[str] = [
    "author_followers",
    "author_following",
    "author_media_count",
    "author_statuses_count",
]
AUTHOR_BOOLEAN: list[str] = ["author_is_verified", "author_is_blue_verified"]

SECTION_STYLE = """
<style>
.section-header {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-radius: 0.6rem;
    background: linear-gradient(90deg, rgba(59,130,246,0.18), rgba(59,130,246,0.05));
    border: 1px solid rgba(59,130,246,0.3);
    margin-top: 2.2rem;
}
.section-header:first-of-type {
    margin-top: 1.4rem;
}
.section-header-text {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.section-header-title {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 600;
}
.section-header-sub {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.85;
}
.section-divider {
    margin: 1.2rem 0 0;
    height: 1px;
    width: 100%;
    background: linear-gradient(90deg, rgba(59,130,246,0.45), rgba(59,130,246,0));
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_tweets(path_str: str) -> pl.DataFrame:
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for json_file in sorted(path.glob("*.json")):
        try:
            frames.append(pl.read_json(json_file, infer_schema_length=None))
        except (PolarsError, FileNotFoundError):
            continue
    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames, how="diagonal_relaxed")
    return df.unique(subset="id")


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> pl.DataFrame:
    raw = load_tweets(path_str)
    if raw.is_empty():
        return raw
    return enrich_frame(raw)


def enrich_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    exprs: list[pl.Expr] = []
    if "createdAt" in df.columns:
        exprs.append(
            pl.col("createdAt")
            .str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False)
            .alias("created_at")
        )
        exprs.append(
            pl.col("createdAt")
            .str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False)
            .dt.date()
            .alias("date")
        )
    if exprs:
        df = df.with_columns(exprs)

    if "author" in df.columns:
        df = df.with_columns(
            [
                pl.col("author").struct.field("userName").cast(pl.Utf8).alias("author_username"),
                pl.col("author").struct.field("name").cast(pl.Utf8).alias("author_name"),
                pl.col("author").struct.field("profilePicture").cast(pl.Utf8).alias("author_profile_image"),
                pl.col("author").struct.field("followers").cast(pl.Float64).alias("author_followers"),
                pl.col("author").struct.field("following").cast(pl.Float64).alias("author_following"),
                pl.col("author").struct.field("mediaCount").cast(pl.Float64).alias("author_media_count"),
                pl.col("author").struct.field("statusesCount").cast(pl.Float64).alias("author_statuses_count"),
                pl.col("author").struct.field("isVerified").cast(pl.Boolean).alias("author_is_verified"),
                pl.col("author").struct.field("isBlueVerified").cast(pl.Boolean).alias("author_is_blue_verified"),
            ]
        )
    return df


def aggregate_daily(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    metrics = [metric for metric in BASE_METRICS if metric in df.columns]
    agg_exprs: list[pl.Expr] = []
    for metric in metrics:
        agg_exprs.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).sum().alias(f"{metric}_sum"),
            ]
        )
    agg_exprs.append(pl.len().alias("tweet_count"))

    for col in AUTHOR_NUMERIC:
        if col in df.columns:
            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
    for col in AUTHOR_BOOLEAN:
        if col in df.columns:
            agg_exprs.append(pl.col(col).cast(pl.Float64).mean().alias(f"{col}_rate"))

    return df.group_by("date").agg(agg_exprs).sort("date")


def humanize(column: str) -> str:
    label = column.replace("_sum", " (sum)").replace("_mean", " (mean)")
    label = label.replace("tweet_count", "Tweet Count")
    label = label.replace("author_", "Author ")
    label = label.replace("viewCount", "View Count")
    label = label.replace("likeCount", "Like Count")
    label = label.replace("retweetCount", "Retweet Count")
    label = label.replace("replyCount", "Reply Count")
    label = label.replace("quoteCount", "Quote Count")
    label = label.replace("bookmarkCount", "Bookmark Count")
    label = label.replace("is verified", "Verified Share")
    label = label.replace("is blue verified", "Blue Verified Share")
    return label.replace("_", " ").title()


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame()
    pdf = df.to_pandas()
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"])
    return pdf


def section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p class='section-header-sub'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
<div class='section-header'>
  <div class='section-header-text'>
    <h3 class='section-header-title'>{title}</h3>
    {subtitle_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_divider() -> None:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Twitter Tweet Analysis", layout="wide")
    st.title("Twitter Tweet Analysis Dashboard")
    st.caption("Investigate engagement for tweets mentioning \"general strike\".")
    st.markdown(SECTION_STYLE, unsafe_allow_html=True)

    data_dir = st.sidebar.text_input("Data directory", value=str(DATA_DIR))
    df = load_dataset(data_dir)
    if df.is_empty():
        st.warning("No JSON tweet data found in this directory.")
        st.stop()
    if "date" not in df.columns:
        st.error("Unable to parse tweet timestamps in the current dataset.")
        st.stop()

    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    language_options = []
    if "lang" in df.columns:
        language_options = list(df["lang"].drop_nulls().unique().sort())
    default_languages: list[str] = []
    selected_languages = st.sidebar.multiselect(
        "Languages",
        options=language_options,
        default=default_languages,
        help="Leave empty to include all languages.",
    )

    keyword_query: str = ""
    keyword_mode: str = "Any"
    min_likes: int = 0
    min_retweets: int = 0
    min_views: int = 0
    min_followers: int = 0
    verified_only: bool = False
    blue_only: bool = False

    with st.sidebar.expander("Advanced filters", expanded=False):
        if "text" in df.columns:
            keyword_query = st.text_input(
                "Keyword search",
                placeholder="strike, italy",
            )
            keyword_mode = st.radio(
                "Keyword match",
                options=["Any", "All"],
                horizontal=True,
            )
        if "likeCount" in df.columns:
            min_likes = int(
                st.number_input(
                    "Minimum likes",
                    min_value=0,
                    value=0,
                    step=10,
                )
            )
        if "retweetCount" in df.columns:
            min_retweets = int(
                st.number_input(
                    "Minimum retweets",
                    min_value=0,
                    value=0,
                    step=5,
                )
            )
        if "viewCount" in df.columns:
            min_views = int(
                st.number_input(
                    "Minimum views",
                    min_value=0,
                    value=0,
                    step=100,
                )
            )
        if "author_followers" in df.columns:
            min_followers = int(
                st.number_input(
                    "Minimum author followers",
                    min_value=0,
                    value=0,
                    step=100,
                )
            )
        if "author_is_verified" in df.columns:
            verified_only = st.checkbox("Only verified authors", value=False)
        if "author_is_blue_verified" in df.columns:
            blue_only = st.checkbox("Only blue-check authors", value=False)

    filtered = df.filter(pl.col("date").is_between(start_date, end_date))
    if selected_languages:
        filtered = filtered.filter(pl.col("lang").is_in(selected_languages))
    if keyword_query and "text" in filtered.columns:
        tokens = [token.strip().lower() for token in re.split(r",|\s+", keyword_query) if token.strip()]
        if tokens:
            text_expr = pl.col("text").cast(pl.Utf8).fill_null("").str.to_lowercase()
            if keyword_mode == "All":
                keyword_expr = pl.lit(True)
                for token in tokens:
                    keyword_expr = keyword_expr & text_expr.str.contains(token, literal=True)
            else:
                keyword_expr = pl.lit(False)
                for token in tokens:
                    keyword_expr = keyword_expr | text_expr.str.contains(token, literal=True)
            filtered = filtered.filter(keyword_expr)
    if min_likes > 0 and "likeCount" in filtered.columns:
        filtered = filtered.filter(pl.col("likeCount").fill_null(0) >= min_likes)
    if min_retweets > 0 and "retweetCount" in filtered.columns:
        filtered = filtered.filter(pl.col("retweetCount").fill_null(0) >= min_retweets)
    if min_views > 0 and "viewCount" in filtered.columns:
        filtered = filtered.filter(pl.col("viewCount").fill_null(0) >= min_views)
    if min_followers > 0 and "author_followers" in filtered.columns:
        filtered = filtered.filter(pl.col("author_followers").fill_null(0) >= min_followers)
    if verified_only and "author_is_verified" in filtered.columns:
        filtered = filtered.filter(pl.col("author_is_verified").fill_null(False))
    if blue_only and "author_is_blue_verified" in filtered.columns:
        filtered = filtered.filter(pl.col("author_is_blue_verified").fill_null(False))
    if filtered.is_empty():
        st.warning("No tweets match the chosen filters.")
        st.stop()

    sortable_metrics = [metric for metric in BASE_METRICS if metric in filtered.columns]

    daily = aggregate_daily(filtered)
    daily_pdf = to_pandas(daily)
    if not daily_pdf.empty:
        daily_pdf = daily_pdf.sort_values("date")

    section_header(
        "Headline Metrics",
        "Quick engagement totals for tweets mentioning \"general strike\".",
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Tweets", f"{filtered.height:,}")
    if "likeCount" in filtered.columns:
        col_b.metric("Total Likes", f"{int(filtered['likeCount'].fill_null(0).sum()):,}")
    if "retweetCount" in filtered.columns:
        col_c.metric("Total Retweets", f"{int(filtered['retweetCount'].fill_null(0).sum()):,}")
    if "viewCount" in filtered.columns:
        avg_views = filtered["viewCount"].fill_null(0).mean()
        col_d.metric("Avg Views/Tweet", f"{avg_views:,.0f}")

    section_divider()

    chart_metric_values: list[str] = []
    if not daily.is_empty():
        other_metrics = [col for col in daily.columns if col not in {"date", "tweet_count"}]
        other_metrics = sorted(other_metrics, key=humanize)
        if "tweet_count" in daily.columns:
            chart_metric_values.append("tweet_count")
        chart_metric_values.extend(other_metrics)
    chart_label_map = {metric: humanize(metric) for metric in chart_metric_values}

    section_header(
        "Daily Metrics Explorer",
        "Bar chart of daily totals or averages for tweets mentioning \"general strike\".",
    )
    if daily_pdf.empty or not chart_metric_values:
        st.info("Not enough daily data to chart metrics.")
    else:
        default_metric = "tweet_count" if "tweet_count" in chart_metric_values else chart_metric_values[0]
        selected_metric = st.selectbox(
            "Metric to graph",
            options=chart_metric_values,
            index=chart_metric_values.index(default_metric),
            format_func=lambda col: chart_label_map.get(col, col),
        )
        chart_df = daily_pdf[["date", selected_metric]].copy()
        chart_df[selected_metric] = chart_df[selected_metric].fillna(0)
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        if getattr(chart_df["date"].dt, "tz", None) is not None:
            chart_df["date"] = chart_df["date"].dt.tz_convert(None)
        y_label = chart_label_map.get(selected_metric, humanize(selected_metric))
        fig = px.bar(
            chart_df,
            x="date",
            y=selected_metric,
            labels={"date": "Date", selected_metric: y_label},
        )
        fig.update_layout(
            margin=dict(l=30, r=10, t=10, b=40),
            yaxis=dict(rangemode="tozero", tickformat=","),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig, config={"responsive": True})

    section_divider()

    section_header(
        "Daily Summary Table",
        "Inspect aggregated metrics for tweets mentioning \"general strike\".",
    )
    if daily_pdf.empty:
        st.info("Daily summary is unavailable for the current filters.")
    else:
        table_pdf = daily_pdf.copy()
        table_pdf["date"] = pd.to_datetime(table_pdf["date"]).dt.date
        rename_map = {
            column: ("Date" if column == "date" else humanize(column))
            for column in table_pdf.columns
        }
        display_daily_pdf = table_pdf.rename(columns=rename_map)
        st.dataframe(display_daily_pdf.set_index("Date"), width="stretch", height=380)
        st.download_button(
            "Download daily summary (CSV)",
            data=display_daily_pdf.to_csv(index=False).encode("utf-8"),
            file_name="daily_summary.csv",
            mime="text/csv",
        )
    section_divider()

    section_header(
        "Day-Level Drilldown",
        "Investigate specific days and standout tweets mentioning \"general strike\".",
    )
    day_options: list = []
    if not daily.is_empty():
        day_options = daily["date"].to_list()
        day_options.sort()
    if day_options:
        default_day = day_options[-1]
        selected_day = st.selectbox(
            "Select a day",
            options=day_options,
            index=day_options.index(default_day),
            format_func=lambda d: d.strftime("%B %d, %Y"),
            key="drilldown_day",
        )
        drill_metric = st.selectbox(
            "Sort metric for the day",
            options=sortable_metrics,
            format_func=humanize,
            key="drilldown_metric",
        ) if sortable_metrics else None
        day_limit = int(
            st.number_input(
                "Tweets to show",
                min_value=1,
                max_value=250,
                value=10,
                step=1,
                format="%d",
                key="drilldown_limit",
            )
        )

        day_filtered = filtered.filter(pl.col("date") == selected_day)
        st.markdown(
            f"{day_filtered.height:,} tweets mentioning \"general strike\" match filters on **{selected_day.strftime('%B %d, %Y')}**."
        )
        if day_filtered.is_empty() or drill_metric is None:
            st.info("No tweets available for the selected day.")
        else:
            day_sorted = (
                day_filtered
                .with_columns(pl.col(drill_metric).fill_null(0).alias("_day_metric"))
                .sort("_day_metric", descending=True)
            )
            detail_cols: list[str] = []
            for col in [
                "date",
                "created_at",
                "text",
                "url",
                "lang",
                "author_username",
                "author_name",
                "author_profile_image",
            ]:
                if col in day_sorted.columns:
                    detail_cols.append(col)
            for col in BASE_METRICS + [
                "bookmarkCount",
                "author_followers",
                "author_following",
                "author_media_count",
                "author_statuses_count",
            ]:
                if col in day_sorted.columns and col not in detail_cols:
                    detail_cols.append(col)
            if drill_metric not in detail_cols and drill_metric in day_sorted.columns:
                detail_cols.append(drill_metric)

            day_selected = day_sorted.select(detail_cols).head(day_limit)
            day_pdf = to_pandas(day_selected)
            if day_pdf.empty:
                st.info("No tweets to display for this day.")
            else:
                if "created_at" in day_pdf.columns:
                    day_pdf["created_at"] = pd.to_datetime(day_pdf["created_at"]).dt.tz_localize(None)
                if "date" in day_pdf.columns:
                    day_pdf["date"] = pd.to_datetime(day_pdf["date"]).dt.date

                top_row = day_pdf.iloc[0]
                st.markdown("**Top tweet of the day**")
                card_cols = st.columns([1, 4])
                if isinstance(top_row.get("author_profile_image"), str) and top_row["author_profile_image"]:
                    card_cols[0].image(top_row["author_profile_image"], width=96)
                else:
                    card_cols[0].markdown("\u00a0")
                author_bits = []
                if isinstance(top_row.get("author_username"), str) and top_row["author_username"]:
                    author_bits.append(f"@{top_row['author_username']}")
                if isinstance(top_row.get("author_name"), str) and top_row["author_name"]:
                    author_bits.append(top_row["author_name"])
                card_cols[1].markdown("**" + " — ".join(author_bits) + "**" if author_bits else "**Author unavailable**")
                metric_val = top_row.get(drill_metric)
                if pd.notna(metric_val):
                    card_cols[1].markdown(f"{humanize(drill_metric)}: {metric_val:,.0f}")
                if isinstance(top_row.get("text"), str):
                    card_cols[1].markdown(f"> {top_row['text']}")
                if isinstance(top_row.get("url"), str) and top_row["url"]:
                    card_cols[1].markdown(f"[Open tweet]({top_row['url']})")

                chart_df = day_pdf.head(min(15, len(day_pdf))).copy()
                if "text" in chart_df.columns:
                    chart_df["label"] = chart_df["text"].fillna("").astype(str).str.slice(0, 80)
                else:
                    chart_df["label"] = chart_df.index.astype(str)
                fig_day = px.bar(
                    chart_df,
                    x=drill_metric,
                    y="label",
                    orientation="h",
                    labels={drill_metric: humanize(drill_metric), "label": "Tweet"},
                )
                fig_day.update_layout(margin=dict(l=10, r=10, t=10, b=40))
                st.plotly_chart(fig_day, config={"responsive": True})

                st.dataframe(day_pdf, width="stretch", hide_index=True)
    else:
        st.info("Add tweets to view day-level drilldowns.")
    section_divider()

    section_header(
        "Top Performing Tweets",
        "Rank tweets mentioning \"general strike\" by the engagement metric that matters most.",
    )
    if not sortable_metrics:
        st.info("No engagement metrics available for ranking.")
    else:
        top_metric = st.selectbox("Sort by", options=sortable_metrics, format_func=humanize)
        top_n = int(st.number_input("Number of tweets", min_value=1, max_value=250, value=15, step=1, format="%d"))

        top_filtered = filtered
        with st.expander("Optional top tweet date filter", expanded=False):
            apply_top_range = st.checkbox(
                "Limit top tweets to a specific date range",
                value=False,
                key="top_range_toggle",
            )
            if apply_top_range:
                top_date_selection = st.date_input(
                    "Top tweets date range",
                    value=(start_date, end_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="top_date_range",
                )
                if isinstance(top_date_selection, tuple):
                    top_start, top_end = top_date_selection
                else:
                    top_start = top_end = top_date_selection
                top_filtered = top_filtered.filter(pl.col("date").is_between(top_start, top_end))
        columns_for_table = [
            "text",
            "date",
            "created_at",
            "lang",
            "author_username",
            "author_name",
            "likeCount",
            "retweetCount",
            "replyCount",
            "quoteCount",
            "viewCount",
            "bookmarkCount",
            "url",
        ]
        available_table_cols = [col for col in columns_for_table if col in top_filtered.columns]
        top_tweets = (
            top_filtered
            .with_columns(pl.col(top_metric).fill_null(0).alias("_sort_metric"))
            .sort("_sort_metric", descending=True)
            .select(available_table_cols)
            .head(top_n)
        )
        top_pdf = to_pandas(top_tweets)
        if not top_pdf.empty:
            if "date" in top_pdf.columns:
                top_pdf["date"] = pd.to_datetime(top_pdf["date"]).dt.date
            if "created_at" in top_pdf.columns:
                top_pdf["created_at"] = pd.to_datetime(top_pdf["created_at"]).dt.tz_localize(None)
            st.dataframe(top_pdf, width="stretch", hide_index=True)
        else:
            st.info("No tweets available for ranking.")

    section_divider()

    section_header(
        "Daily Top Tweet Spotlight",
        "Browse one standout \"general strike\" tweet per day within a month.",
    )
    month_options: list[str] = []
    if "date" in filtered.columns:
        month_df = (
            filtered
            .select(pl.col("date").dt.strftime("%Y-%m").alias("month"))
            .drop_nulls()
            .unique()
            .sort("month")
        )
        month_options = month_df["month"].to_list()

    default_month = "2025-10" if "2025-10" in month_options else (month_options[-1] if month_options else None)
    if month_options and default_month:
        selected_month = st.selectbox(
            "Month",
            options=month_options,
            index=month_options.index(default_month),
            key="carousel_month",
        )
    elif month_options:
        selected_month = st.selectbox("Month", options=month_options, key="carousel_month")
    else:
        selected_month = None

    if sortable_metrics:
        carousel_metric = st.selectbox(
            "Metric used for daily winners",
            options=sortable_metrics,
            format_func=humanize,
            key="carousel_metric",
        )
    else:
        carousel_metric = None

    carousel_data = pl.DataFrame()
    if selected_month:
        selected_year, selected_month_num = map(int, selected_month.split("-"))
        carousel_data = filtered.filter(
            (pl.col("date").dt.year() == selected_year)
            & (pl.col("date").dt.month() == selected_month_num)
        )

    if carousel_data.is_empty() or carousel_metric is None:
        st.info("No tweets available for the selected month.")
    else:
        highlighted = (
            carousel_data
            .with_columns(pl.col(carousel_metric).fill_null(0).alias("_carousel_metric"))
            .sort(["date", "_carousel_metric"], descending=[False, True])
            .unique(subset=["date"], keep="first")
            .sort("date")
            .drop("_carousel_metric")
        )
        simple_cols: list[str] = []
        for col in ["date", "text", "url", "lang", "author_username", "author_name", "author_profile_image"]:
            if col in highlighted.columns:
                simple_cols.append(col)
        for col in BASE_METRICS + ["bookmarkCount", "author_followers", "author_following", "author_media_count", "author_statuses_count"]:
            if col in highlighted.columns and col not in simple_cols:
                simple_cols.append(col)
        if carousel_metric not in simple_cols and carousel_metric in highlighted.columns:
            simple_cols.append(carousel_metric)
        highlighted = highlighted.select(simple_cols)
        spotlight_pdf = to_pandas(highlighted)
        if spotlight_pdf.empty:
            st.info("Unable to determine top tweets for the selected month.")
        else:
            spotlight_pdf["date"] = pd.to_datetime(spotlight_pdf["date"])
            spotlight_pdf = spotlight_pdf.sort_values("date")
            total_days = len(spotlight_pdf)
            carousel_selection = st.slider(
                "Browse daily winners",
                min_value=1,
                max_value=total_days,
                value=1,
                format="Day %d",
            )
            row = spotlight_pdf.iloc[carousel_selection - 1]
            day_label = row["date"].strftime("%B %d, %Y")
            st.markdown(f"#### {day_label}")
            author_username = row.get("author_username")
            author_name = row.get("author_name")
            author_profile_image = row.get("author_profile_image")
            info_cols = st.columns([1, 4])
            if isinstance(author_profile_image, str) and author_profile_image:
                info_cols[0].image(author_profile_image, width=96)
            else:
                info_cols[0].markdown("\u00a0")
            author_parts = []
            if isinstance(author_username, str) and author_username:
                author_parts.append(f"@{author_username}")
            if isinstance(author_name, str) and author_name:
                author_parts.append(author_name)
            if author_parts:
                info_cols[1].markdown("**" + " — ".join(author_parts) + "**")
            else:
                info_cols[1].markdown("**Author unavailable**")
            metric_value = row.get(carousel_metric)
            if pd.notna(metric_value):
                st.markdown(
                    f"**{humanize(carousel_metric)}:** {metric_value:,.0f}"
                )
            if isinstance(row.get("text"), str):
                st.markdown(f"> {row['text']}")
            if isinstance(row.get("url"), str) and row["url"]:
                st.markdown(f"[Open tweet]({row['url']})")

            metric_fields = [
                "likeCount",
                "retweetCount",
                "replyCount",
                "quoteCount",
                "viewCount",
                "bookmarkCount",
            ]
            available_metrics = [field for field in metric_fields if field in spotlight_pdf.columns]
            if available_metrics:
                columns_needed = min(3, len(available_metrics))
                metric_columns = st.columns(columns_needed)
                for idx, field in enumerate(available_metrics):
                    value = row.get(field)
                    display_value = f"{value:,.0f}" if pd.notna(value) else "n/a"
                    metric_columns[idx % columns_needed].metric(
                        humanize(field),
                        display_value,
                    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with Polars for data processing. Use `streamlit run src/streamlit_app.py` to launch locally."
    )


if __name__ == "__main__":
    main()
