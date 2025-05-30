# COMM4190 Final Project - BioEquity Insight

## An AI-Powered Stock Analysis Assistant for Biotechnology Investments

**By Group M** - Vidur Saigal

## Introduction

Throughout this semester, we've explored the transformative potential of Artificial Intelligence (AI) and Large Language Models (LLMs) across various applications. Our journey has culminated in developing BioEquity Insight - an AI-powered stock analysis assistant specifically designed for biotechnology investments.

The biotechnology sector represents one of the most dynamic yet challenging areas for investors. It's characterized by complex scientific developments, regulatory hurdles, and binary events that can dramatically impact stock prices overnight. Our tool bridges the knowledge gap for investors by providing comprehensive analysis, visualization, and insights tailored to the unique nature of biotech stocks.

## BioEquity Insight: Value Proposition

### Background

The biotechnology investment landscape presents unique challenges compared to other sectors. Investors must interpret clinical trial data, understand regulatory pathways, and assess scientific innovations without necessarily having extensive scientific expertise. Traditional stock analysis tools often fall short in providing the specialized context needed for biotech investing decisions.

While platforms like Yahoo Finance and Bloomberg provide general market data, they rarely offer biotech-specific insights such as clinical trial progress, regulatory milestone tracking, or patent expiration implications. The rise of AI capabilities has opened new possibilities for sector-specific investment analysis that integrates multiple data sources and domain knowledge.

BioEquity Insight addresses this gap by combining real-time financial data with biotech-specific catalysts, providing intuitive visualizations, and leveraging AI to translate complex scientific developments into actionable investment insights.

### Key Features

- **Biotech-Focused Stock Analysis**: Specialized prompting and data collection focused on metrics most relevant to biotechnology companies
- **Catalyst Tracking**: Automated identification and analysis of key biotech catalysts including FDA decisions, clinical trial results, and patent developments
- **Multi-Stock Comparison**: Side-by-side analysis of multiple biotech stocks to identify sector trends and comparative advantages
- **Dynamic Visualization**: AI-generated charts that highlight the metrics most relevant to each specific user query
- **Plain-Language Insights**: Complex biotech developments explained in accessible terms for investors of all expertise levels

## Technology Stack

BioEquity Insight leverages several powerful technologies:

- **Financial Data**: Real-time and historical stock information via Yahoo Finance API
- **News & Catalysts**: Custom Google Search integration for recent biotech-specific news
- **Natural Language Processing**: OpenAI's language models for query interpretation and analysis
- **Interactive Visualization**: Custom Plotly charts generated based on specific user questions
- **LangChain Framework**: For orchestrating the AI components and managing prompt workflows

## How It Works

1. **Query Processing**: The system extracts stock tickers from natural language user queries
2. **Data Collection**: Comprehensive financial information is gathered for each ticker
3. **Catalyst Research**: Recent news and events are collected for relevant biotech companies
4. **AI Analysis**: The system generates insights by analyzing both financial data and recent catalysts
5. **Visualization**: Custom charts are created to illustrate key points relevant to the user's query
6. **Results Presentation**: Analysis and visualizations are presented alongside source information

## Use Cases

- **Pipeline Analysis**: Evaluate a biotech company's drug development pipeline and milestone timeline
- **Regulatory Impact Assessment**: Analyze how FDA decisions have historically affected similar stocks
- **Comparative Analysis**: Compare multiple biotech companies within specific therapeutic areas
- **Catalyst Prediction**: Identify upcoming events likely to impact biotech stock performance
- **Technical Analysis**: Visualize biotech-specific technical indicators and trading patterns

## Future Development

- Integration with clinical trial databases for real-time trial updates
- Patent expiration tracking and impact analysis
- Biotech-specific sentiment analysis from scientific publications
- Enhanced visualization of drug pipeline progress
- Integration with genomic databases for scientific breakthrough assessment

---

*BioEquity Insight is a project created for COMM4190 (Spring 2025) and is intended for educational purposes only. The financial analysis should not be considered investment advice.*


# BioEquity Insight - Setup Guide

Below are instructions for setting up and running the BioEquity Insight application.

## Prerequisites

- Python 3.8 or higher
- Access to the following APIs:
  - OpenAI API key
  - Google API key
  - Google Custom Search Engine ID

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/mbod/comm4190_S25_Final_Project_GroupM.git
cd comm4190_S25_Final_Project_GroupM
```

### 2. Create a Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install langchain langchain-openai yfinance requests matplotlib mplfinance pandas seaborn plotly python-dotenv
```

### 4. Set Up API Keys

Create a .env file in the project root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_custom_search_engine_id
```

To obtain these keys:
- **OpenAI API Key**: Sign up at [OpenAI](https://openai.com/api/)
- **Google API Key**: Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
- **Google Custom Search ID**: Set up a Custom Search Engine at [Google Programmable Search](https://programmablesearchengine.google.com/)

### 5. Running the Application

The application can be run in two ways:

#### Option 1: Using the ChainLit Interface

For the interactive web interface:

```bash
cd AppFiles
chainlit run app.py
```

This will launch a local web server and open the application in your default web browser.

#### Option 2: Using the Python Script Directly

To use the analytical functions directly in Python:

```python
from AppFiles.script import answer_user_query

# Example usage
result = answer_user_query("How is MRNA stock performing compared to PFE?")
print(result["text"])

# If you want to display the chart in a notebook
import plotly.io as pio
pio.show(result["chart_figure"])
```

## Example Queries

Here are some example queries to try with BioEquity Insight:

- "Analyze recent clinical trial results for Moderna"
- "Compare the financial performance of Amgen and BIIB over the past year."
- "How did the FDA approval affect Regeneron's stock price?"
- "What are the upcoming catalysts for GILD?"
- "Compare the pipeline strength of VRTX and BMRN"

## Troubleshooting

- **API Rate Limits**: If you encounter errors related to API limits, wait a few minutes before trying again.
- **Missing Data**: Some biotech stocks may have incomplete data depending on their size and reporting history.

---

For more information or help with this project, please contact the project maintainers or refer to the documentation in the repository.
