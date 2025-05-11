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

