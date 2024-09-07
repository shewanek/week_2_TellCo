# TellCo Analysis Project

This project analyzes telecommunication data for the potential purchase of TellCo in the Republic of Pefkakia. It includes exploratory data analysis, visualizations, and a Streamlit dashboard.

Below is a structured `README.md` that covers both **Task 1** (User Overview Analysis) and **Task 2** (User Engagement Analysis). It explains the goals, requirements, and usage of both tasks.

---

This project involves the analysis of telecom data for TellCo, a mobile service provider. The analysis is performed in two tasks:

- **Task 1**: User Overview Analysis
- **Task 2**: User Engagement Analysis

The goal is to provide insights into user behavior, network performance, and engagement, helping in making data-driven decisions for business growth.

---

## Project Structure

```
├── scripts
│   ├── conect_db.py        # Task 1: Database connection utility
│   ├── tellCo_analysis.py        # Task 1: Functions for user overview analysis
│   ├── user_engag_analysis.py  # Task 2: Class-based implementation for engagement analysis
├── notebooks
│   ├── user_overview_EDA.ipynb          # Task 1: User overview EDA
│   ├── user_engag_EDA.ipynb  # Task 2: User engagement analysis
├── README.md                   # This file
└── requirements.txt            # Project dependencies
```

---

## Prerequisites

- Python 3.8+
- PostgreSQL database with telecom data
- Required libraries: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `SQLAlchemy`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Task 1: User Overview Analysis

**Goal**: To analyze the overall user behavior, including top handset types, manufacturers, and user behavior with various applications.

### Steps

1. **Database Connection**: 
   - Use `conect_db.py` to establish a connection to the PostgreSQL database.
   
2. **User Overview Analysis**:
   - In `tellCo_analysis.py`, several functions are defined to explore the top handsets, manufacturers, and user traffic.
   
3. **Perform EDA**:
   - Open `user_overview_EDA.ipynb`, import the functions from `data_analysis.py`, and perform the analysis on the dataset.
   
### Key Outputs
- Top 10 handsets and manufacturers
- User behavior and distribution of traffic across applications
- Data visualizations highlighting key trends in usage

---

## Task 2: User Engagement Analysis

**Goal**: To track user engagement using metrics such as session frequency, session duration, and total traffic (download/upload), and classify users based on engagement levels.

### Steps

1. **Aggregate User Metrics**:
   - Use the `aggregate_user_metrics` method in `user_engag_analysis.py` to calculate session frequency, session duration, and total traffic per user (MSISDN).
   
2. **Normalize Metrics**:
   - The `normalize_metrics` method scales the data for clustering.
   
3. **K-means Clustering**:
   - The `run_kmeans` method clusters users into three groups based on engagement metrics.
   - Use the `plot_elbow_method` method to determine the optimal number of clusters.
   
4. **Application Usage**:
   - Aggregate user traffic per application using the provided method and visualize the top 3 most used applications.
   
5. **Visualizations**:
   - Use `matplotlib` to visualize the top apps and clustering results in `user_engagement_EDA.ipynb`.

### Key Outputs
- Top 10 users by engagement metrics (session frequency, duration, traffic)
- K-means clustering of users into 3 groups
- Visualizations of top 3 most used applications

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/shewanek/week_2_TellCo.git
   cd week_2_TellCo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebooks:
   - For Task 1 (User Overview Analysis), open `user_EDA.ipynb`.
   - For Task 2 (User Engagement Analysis), open `user_engagement_EDA.ipynb`.

---

## File Details

- **scripts/db_connection.py**: Connects to the PostgreSQL database.
- **scripts/data_analysis.py**: Contains functions for aggregating user data and performing EDA for Task 1.
- **scripts/user_engagement_analysis.py**: A class-based script for Task 2 to analyze user engagement using K-means clustering and visualizations.
- **notebooks/user_EDA.ipynb**: Notebook performing the user overview analysis for Task 1.
- **notebooks/user_engagement_EDA.ipynb**: Notebook performing the user engagement analysis for Task 2.

---

## Contact

For any questions or issues, please contact [zshewanek@gmail.com](mailto:zshewanek@gmail.com).

---