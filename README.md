
# Cross-Partisan Interaction Analysis on Twitter/X

## Overview

This project aims to analyze cross-partisan interactions on the social media platform Twitter/X, focusing on how users from different political ideologies interact with each other. The analysis includes topic classification, sentiment analysis, entity recognition, and ideological clustering, with the ultimate goal of understanding the dynamics of political discourse online.

Key features of the project include:
- **Topic Analysis**: Identifying and categorizing the main topics discussed in cross-partisan interactions.
- **Sentiment Analysis**: Classifying the sentiment of tweets (positive, negative, neutral) to assess the emotional tone of cross-partisan interactions.
- **Entity Recognition**: Detecting and categorizing entities (e.g., people, organizations) mentioned in tweets.
- **Ideological Clustering**: Using a Gaussian Mixture Model (GMM) to classify users based on their political leanings.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Zhenyi-Q/Cross-Partisan-Interaction-Analysis-on-Twitter-X.git
    cd Cross-Partisan-Interaction-Analysis-on-Twitter-X
    ```

2. **Set Up a Virtual Environment** (optional but recommended):
    ```bash
    conda create --name cpi-analysis python=3.8
    conda activate cpi-analysis
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Necessary Datasets**:
    - Ensure that the required datasets (e.g., `user-ideal-points-201807.csv`, Twitter data) are placed in the `./data/` directory.

## Usage

### 1. Topic Analysis

To perform topic analysis on the collected Twitter data:
```python
from topic_analysis import run_topic_analysis

run_topic_analysis(data_path="./data/tweets.csv")
