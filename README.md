# Twitter Search Analysis

A Python 3.13 project for analyzing large JSON files containing thousands of tweets. Uses [uv](https://github.com/astral-sh/uv) for environment management, [polars](https://www.pola.rs/) for fast data analysis, and [seaborn](https://seaborn.pydata.org/) for visualization.

## Features
- Efficiently loads and analyzes large tweet datasets
- Data exploration and visualization
- Ready for custom analysis once data is added

## Setup Instructions

1. **Install uv** (if not already installed):
   ```sh
   pip install uv
   ```
2. **Create environment and install dependencies:**
   ```sh
   uv venv .venv
   uv pip install polars seaborn
   ```
3. **Activate environment:**
   ```sh
   source .venv/bin/activate
   ```
4. **Run analysis:**
   ```sh
   python src/analyze.py
   ```

## Data
Add your JSON tweet files to the `data/` directory. Update the analysis script as needed to match your data format.

## Reports
`reports/` holds the October 2025 graph exports: daily tweet, like, retweet, view and author-follower charts for tweets mentioning "general strike", and a summary comparing the periods before and after 2025-10-16.
