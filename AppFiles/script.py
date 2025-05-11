import langchain
from langchain_openai import ChatOpenAI  # Changed to use ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import yfinance as yf
import requests
import datetime as dt
import matplotlib.pyplot as plt
import io
import base64
import mplfinance
import pandas as pd
from typing import Any
import seaborn as sns
sns.set_theme()  
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
GOOGLE_CX = os.getenv("GOOGLE_CX")


def get_stock_data(ticker: str) -> tuple:
    """Retrieve basic stock info for the given ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = (
        f"Name: {info.get('longName', 'N/A')}\n"
        f"Short Name: {info.get('shortName', 'N/A')}\n"
        f"Symbol: {info.get('symbol', ticker)}\n"
        f"Sector: {info.get('sector', 'N/A')}\n"
        f"Industry: {info.get('industry', 'N/A')}\n"
        f"Currency: {info.get('currency', 'N/A')}\n"
        f"Exchange: {info.get('exchange', 'N/A')}\n"
        f"Current Price: {info.get('regularMarketPrice', 'N/A')}\n"
        f"Open: {info.get('open', 'N/A')}\n"
        f"Previous Close: {info.get('previousClose', 'N/A')}\n"
        f"Day's Low/High: {info.get('dayLow', 'N/A')} / {info.get('dayHigh', 'N/A')}\n"
        f"52-Week Low/High: {info.get('fiftyTwoWeekLow', 'N/A')} / {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        f"Volume: {info.get('volume', 'N/A')}\n"
        f"Average Volume: {info.get('averageVolume', 'N/A')}\n"
        f"Market Cap: {info.get('marketCap', 'N/A')}\n"
        f"Beta: {info.get('beta', 'N/A')}\n"
        f"PE Ratio (TTM): {info.get('trailingPE', 'N/A')}\n"
        f"EPS (TTM): {info.get('trailingEps', 'N/A')}\n"
        f"Dividend Rate: {info.get('dividendRate', 'N/A')}\n"
        f"Dividend Yield: {info.get('dividendYield', 'N/A')}\n"
        f"Ex-Dividend Date: {info.get('exDividendDate', 'N/A')}\n"
        f"Earnings Timestamp: {info.get('earningsTimestamp', 'N/A')}\n"
        f"1-Year Target Estimate: {info.get('targetMeanPrice', 'N/A')}\n"
        f"Fifty Day Average: {info.get('fiftyDayAverage', 'N/A')}\n"
        f"Two Hundred Day Average: {info.get('twoHundredDayAverage', 'N/A')}\n"
        f"Long Business Summary: {info.get('longBusinessSummary', 'N/A')}\n"
    )

    # Retrieve historical data from inception up to yesterday
    # This ensures only complete days of data are included
    try:
        # Get today's date
        today = dt.date.today()
        
        # Find the most recent business day
        # If today is Sunday (6) or Saturday (5)
        if today.weekday() == 6:  # Sunday
            days_to_subtract = 2  # Go back to Friday
        elif today.weekday() == 5:  # Saturday
            days_to_subtract = 1  # Go back to Friday
        else:
            days_to_subtract = 0  # Use today
            
        # Get the most recent business day
        most_recent_business_day = today - dt.timedelta(days=days_to_subtract)
        
        print(f"Fetching complete historical data for {ticker}")
        
        # Get historical data using period="max" to fetch all available history
        historical = stock.history(period="max")
        
        if not historical.empty:
            summary += f"\nHistorical Data available from {historical.index[0].strftime('%Y-%m-%d')} to {historical.index[-1].strftime('%Y-%m-%d')}\n"
            summary += "Last 5 trading days:\n"
            summary += historical.tail(5).to_string() + "\n"
        else:
            # Fallback to period-based query for 1 year
            historical = stock.history(period="1y")
            summary += "\nUsing period-based historical data (last 5 trading days):\n"
            summary += historical.tail(5).to_string() + "\n"
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        historical = pd.DataFrame()  # Empty DataFrame as fallback
        summary += "\nCould not retrieve historical data for this ticker.\n"
    
    return summary, historical


def get_recent_catalysts(company_names, api_key: str, cx: str) -> dict:
    """
    Use Google Custom Search API to find recent catalysts for the given companies.
    
    Args:
        company_names: Either a single company name (str) or list of company names
        api_key: Google API key
        cx: Google Custom Search Engine ID
        
    Returns:
        Dictionary mapping company names to their catalysts information
    """
    # Convert single company name to list for consistent processing
    if isinstance(company_names, str):
        company_names = [company_names]
    
    results = {}
    
    for company_name in company_names:
        search_query = f"{company_name} recent stock catalysts, FDA approvals, clinical trials, significant news"
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "q": search_query,
            "key": api_key,
            "cx": cx,
            "num": 5  # Limit to 5 results for brevity
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()

            # Extract relevant information from the search results
            catalysts = []
            for item in search_results.get("items", []):
                title = item.get("title", "No title")
                link = item.get("link", "No link")
                snippet = item.get("snippet", "No snippet")
                catalysts.append(f"- {title}\n  {snippet}\n  Link: {link}")

            if not catalysts:
                results[company_name] = "No recent catalysts found for this company."
            else:
                results[company_name] = "\n\n".join(catalysts)
                
        except Exception as e:
            results[company_name] = f"Error fetching catalysts: {str(e)}"
    
    return results

# Define the prompt template to extract multiple tickers
ticker_prompt_template = (
    "You are a helpful financial assistant. "
    "Extract the stock ticker symbols (uppercase, 1-5 letters) from the user's request.\n"
    "If the user is comparing or asking about multiple companies, return ALL relevant ticker symbols, separated by commas.\n"
    "If only one ticker is mentioned, just return that single ticker.\n"
    "Only return the ticker symbols, nothing else.\n"
    "For example, if the prompt mentions 'Compare Merck and Pfizer', output 'MRK,PFE'. "
    "For 'How is Merck doing?', output 'MRK'.\n\n"
    "User Prompt: {user_prompt}\n"
    "Tickers:"
)

ticker_prompt = PromptTemplate(
    input_variables=["user_prompt"],
    template=ticker_prompt_template
)

# Initialize using ChatOpenAI instead of OpenAI
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,  # Use environment variable
    model="o3-mini",  # Using the standard chat model
)

# Create the ticker extraction chain
ticker_chain = LLMChain(llm=llm, prompt=ticker_prompt)

def extract_tickers_from_prompt(user_prompt: Any) -> list:
    """Extract one or more ticker symbols from the user prompt."""
    # If Chainlit passed a Message, pull out .content, otherwise assume it's already a str
    prompt_text = user_prompt.content if hasattr(user_prompt, "content") else user_prompt
    print("RAW PROMPT ►", repr(prompt_text))
    
    # Get comma-separated ticker string from LLM
    tickers_str = ticker_chain.run(user_prompt=prompt_text).strip()
    print("RAW TICKERS ►", repr(tickers_str))
    
    # Split by comma and clean each ticker
    tickers = [ticker.strip().upper() for ticker in tickers_str.split(',')]
    
    # Remove any empty strings
    tickers = [ticker for ticker in tickers if ticker]
    
    print(f"Extracted {len(tickers)} ticker(s): {tickers}")
    return tickers

# Update the analysis prompt template to be more concise
analysis_prompt_template = (
    "You are a financial analyst. Below is the Yahoo Finance data for the stock and recent catalysts:\n\n"
    "### Stock Data:\n{stock_data}\n\n"
    "### Recent Catalysts:\n{catalysts}\n\n"
    "Based on all this information, answer the following question, but with smart, accurate, numerical analysis and well-thought out takes. Include insights from both the stock data and catalysts in your analysis:\n"
    "{user_prompt}\n"
)

analysis_prompt = PromptTemplate(
    input_variables=["stock_data", "catalysts", "user_prompt"],
    template=analysis_prompt_template
)

# Create the analysis chain.
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

def answer_user_query(user_prompt: str) -> dict:
    try:
        # Step 1: Extract ticker(s) automatically using the LLM
        tickers = extract_tickers_from_prompt(user_prompt)
        print(f"Extracted Tickers: {tickers}")
        
        if not tickers:
            return {"text": "I couldn't identify any stock tickers in your request. Please mention specific companies or stock symbols."}
        
        # Step 2: Collect data for each ticker
        all_stock_data = {}
        all_historical_data = {}
        company_names = {}
        
        for ticker in tickers:
            # Get stock data for this ticker
            stock_data, historical_data = get_stock_data(ticker)
            all_stock_data[ticker] = stock_data
            all_historical_data[ticker] = historical_data
            
            # Extract company name from the stock data
            lines = stock_data.split('\n')
            company_name = lines[0].replace("Name: ", "").strip() if lines else ticker
            company_names[ticker] = company_name
            
            # Trim business summaries to reduce token count
            for i, line in enumerate(lines):
                if line.startswith("Long Business Summary:"):
                    summary_text = line.replace("Long Business Summary: ", "")
                    if len(summary_text) > 300:
                        lines[i] = f"Long Business Summary: {summary_text[:300]}..."
                    break
            all_stock_data[ticker] = '\n'.join(lines)

        # Step 3: Fetch recent catalysts for all companies using Google Custom Search
        all_catalysts = get_recent_catalysts(list(company_names.values()), 
                                            GOOGLE_API_KEY,  # Use environment variable 
                                            GOOGLE_CX)      # Use environment variable
        
        # Step 4: Update the analysis prompt to handle multiple companies
        if len(tickers) > 1:
            # For multiple tickers, use a comparative analysis prompt
            comparative_prompt = create_comparative_analysis_prompt(
                tickers, all_stock_data, all_catalysts, company_names, user_prompt
            )
            result = llm.invoke(comparative_prompt).content
        else:
            # For a single ticker, use the existing analysis chain
            ticker = tickers[0]
            result = analysis_chain.run(
                stock_data=all_stock_data[ticker],
                catalysts=all_catalysts[company_names[ticker]],
                user_prompt=user_prompt
            )

        # Step 5: Generate chart scripts for visualization
        chart_script = None
        chart_figure = None
        
        if len(tickers) > 1:
            # Generate a comparative chart for multiple tickers
            chart_script = generate_comparative_chart_script(
                tickers, all_stock_data, all_historical_data, user_prompt
            )
        else:
            # Use existing function for single ticker
            ticker = tickers[0]
            chart_script = generate_chart_script(
                all_stock_data[ticker], ticker, all_historical_data[ticker], user_prompt
            )
            
        if chart_script:
            if len(tickers) > 1:
                # Execute the comparative chart script
                chart_figure = execute_comparative_chart_script(
                    chart_script, all_historical_data, tickers
                )
            else:
                # Use existing function for single ticker
                ticker = tickers[0]
                chart_figure = execute_chart_script(
                    chart_script, all_historical_data[ticker], ticker
                )
        
        # Step 6: Combine the analysis result with the sources
        sources_text = []
        for ticker, company_name in company_names.items():
            if company_name in all_catalysts:
                sources_text.append(f"### Sources for {company_name} ({ticker}):\n{all_catalysts[company_name]}")
        
        final_response = f"### Analysis:\n{result}\n\n" + "\n\n".join(sources_text)
        
        return {
            "text": final_response,
            "chart_figure": chart_figure,
            "script": chart_script
        }
        
    except Exception as e:
        import traceback
        print(f"Error in answer_user_query: {str(e)}")
        print(traceback.format_exc())
        return {"text": f"An error occurred while processing your request: {str(e)}"}

# Add this new function to generate chart scripts using OpenAI
def generate_chart_script(stock_data: str, ticker: str, historical_data: pd.DataFrame, user_prompt: str) -> str:
    """
    Ask OpenAI to generate a matplotlib/plotly chart script for the given stock data.
    """
    # Create enhanced prompt for OpenAI to generate a visualization script
    chart_prompt = f"""
    Create a Python function that generates a visualization specifically addressing this user question:
    "{user_prompt}"
    
    Ticker: {ticker}
    
    Stock Summary Data:
    {stock_data}
    
    The historical data is available as a pandas DataFrame called 'historical_data' with the following structure:
    {historical_data.head().to_string()}
    
    Additional data is provided in the 'data_dict' parameter, containing:
    - ticker_info: Dictionary with all Yahoo Finance data (access with data_dict['ticker_info'])
    - key_metrics: Dictionary with important metrics like PE ratio, market cap (access with data_dict['key_metrics'])
    - dividend_history: Pandas Series with dividend history (access with data_dict['dividend_history'])
    - dividend_df: DataFrame with 'Dividend' column (access with data_dict['dividend_df'])
    - income_statement: Income statement DataFrame (access with data_dict['income_statement'])
    - balance_sheet: Balance sheet DataFrame (access with data_dict['balance_sheet'])
    - cash_flow: Cash flow DataFrame (access with data_dict['cash_flow'])
    - recommendations: Analyst recommendations DataFrame (access with data_dict['recommendations'])
    - major_holders, institutional_holders: Ownership DataFrames
    
    IMPORTANT NOTES:
    1. When comparing dates, all datetime indexes have been normalized to have no timezone information.
       If you're performing date comparisons or filtering, use naive datetime objects without timezone info.
    2. For pandas date frequency strings, use 'YE' instead of 'Y', and 'ME' instead of 'M' for year-end and month-end.
    3. For date handling, use the datetime module which is available as 'dt' in your function.
    
    Write a function called 'generate_chart' that:
    1. Takes historical_data as the first parameter, and data_dict as the second parameter
    2. Creates an informative visualization using Plotly (import plotly.graph_objects as go)
    3. Shows metrics and trends SPECIFICALLY RELEVANT to the user's question above
    4. Carefully checks if keys exist in data_dict and DataFrames aren't empty before using them
    5. Returns a plotly.graph_objects.Figure object
    
    Return ONLY the Python code without any explanation.
    """
    
    # Use the same LLM we already have
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    chart_prompt_template = PromptTemplate(
        input_variables=["chart_prompt"],
        template="{chart_prompt}"
    )
    
    chart_chain = LLMChain(llm=llm, prompt=chart_prompt_template)
    
    # Get the chart generation script
    try:
        chart_script = chart_chain.run(chart_prompt=chart_prompt).strip()
        print("=== Generated Chart Script ===")
        print(chart_script)
        print("=== End Chart Script ===\n")
        return chart_script
    except Exception as e:
        print(f"Error generating chart script: {e}")

# Function to execute the generated script safely and get chart as base64
def execute_chart_script(script: str, historical_data: pd.DataFrame, ticker: str = None) -> object:
    """Execute the generated chart script and return a Plotly figure."""
    try:
        # Create a namespace for execution that includes Plotly and core libraries
        namespace = {
            'pd': pd,
            'plt': plt,
            'historical_data': historical_data.copy(),
            'np': __import__('numpy'),
            'go': go,
            'make_subplots': make_subplots,
            'plotly': __import__('plotly'),
            'ticker': ticker,
            'dt': dt,  # Add datetime module as dt
            'datetime': dt,  # Also add as datetime for flexibility
        }
        
        # Normalize the datetime index by removing timezone information
        if hasattr(namespace['historical_data'].index, 'tz') and namespace['historical_data'].index.tz is not None:
            print("Normalizing historical_data timezone from", namespace['historical_data'].index.tz)
            namespace['historical_data'].index = namespace['historical_data'].index.tz_localize(None)
        
        # Add comprehensive stock data if ticker is provided
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                
                # Store all ticker info in a structured format
                namespace['ticker_info'] = stock.info
                
                # Extract key metrics from ticker_info for easier access
                metrics = {}
                key_fields = [
                    'regularMarketPrice', 'previousClose', 'open', 'dayLow', 'dayHigh',
                    'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'volume', 'averageVolume',
                    'marketCap', 'beta', 'trailingPE', 'forwardPE', 'trailingEps',
                    'dividendRate', 'dividendYield', 'exDividendDate',
                    'fiftyDayAverage', 'twoHundredDayAverage',
                    'shortRatio', 'profitMargins', 'operatingMargins'
                ]
                
                for field in key_fields:
                    if field in stock.info:
                        metrics[field] = stock.info[field]
                
                namespace['key_metrics'] = metrics
                
                # Store dividend history (Series with DatetimeIndex) and normalize timezone
                dividend_history = stock.dividends
                if not dividend_history.empty and hasattr(dividend_history.index, 'tz') and dividend_history.index.tz is not None:
                    dividend_history.index = dividend_history.index.tz_localize(None)
                namespace['dividend_history'] = dividend_history
                
                # Convert dividend Series to DataFrame (simpler to use)
                if not dividend_history.empty:
                    div_df = dividend_history.to_frame('Dividend')
                    div_df.index.name = 'Date'
                    namespace['dividend_df'] = div_df
                else:
                    namespace['dividend_df'] = pd.DataFrame()
                
                # Normalize timezone for financial statements if needed
                for financial_stmt, stmt_name in [
                    (stock.income_stmt, 'income_statement'),
                    (stock.balance_sheet, 'balance_sheet'),
                    (stock.cash_flow, 'cash_flow')
                ]:
                    if not financial_stmt.empty and hasattr(financial_stmt.columns, 'tz') and financial_stmt.columns.tz is not None:
                        financial_stmt.columns = financial_stmt.columns.tz_localize(None)
                    namespace[stmt_name] = financial_stmt
                
                # Additional useful data
                recommendations = stock.recommendations
                if not recommendations.empty and hasattr(recommendations.index, 'tz') and recommendations.index.tz is not None:
                    recommendations.index = recommendations.index.tz_localize(None)
                namespace['recommendations'] = recommendations
                
                namespace['calendar'] = stock.calendar
                
                # Include major holders when available
                try:
                    namespace['major_holders'] = stock.major_holders
                    namespace['institutional_holders'] = stock.institutional_holders
                except:
                    pass
                
                # Print data availability for debugging
                print(f"Data provided to chart function:")
                for key, value in namespace.items():
                    if key not in ['pd', 'plt', 'np', 'go', 'make_subplots', 'plotly']:
                        data_type = type(value).__name__
                        data_size = "N/A" 
                        if hasattr(value, "shape"):
                            data_size = value.shape
                        elif isinstance(value, dict):
                            data_size = f"{len(value)} keys"
                        print(f"- {key}: {data_type} ({data_size})")
                        
            except Exception as e:
                print(f"Failed to add financial data: {e}")
                # Set defaults for missing data
                namespace['ticker_info'] = {}
                namespace['key_metrics'] = {}
                namespace['dividend_history'] = pd.Series()
                namespace['dividend_df'] = pd.DataFrame()
                namespace['income_statement'] = pd.DataFrame()
                namespace['balance_sheet'] = pd.DataFrame() 
                namespace['cash_flow'] = pd.DataFrame()
                
        # Execute the script in the namespace
        exec(script, namespace) 
        
        # Check if generate_chart function was created
        if 'generate_chart' not in namespace:
            print("Error: generate_chart function not found in script")
            return None

        # Get function signature to determine required parameters
        import inspect
        sig = inspect.signature(namespace['generate_chart'])
        param_count = len(sig.parameters)
        param_names = list(sig.parameters.keys())
        
        print(f"Chart function expects {param_count} parameters: {param_names}")
        
        # Call with appropriate number of parameters
        if param_count == 1:
            fig = namespace['generate_chart'](historical_data)
        else:
            # Pass historical_data as first arg and dict of all other data as second arg
            data_dict = {k: v for k, v in namespace.items() 
                        if k not in ['pd', 'plt', 'np', 'go', 'make_subplots', 'plotly', 'generate_chart']}
            try:
                fig = namespace['generate_chart'](historical_data, data_dict)
            except Exception as e:
                print(f"Error with data dictionary: {e}")
                # Fallback to simpler approach
                if param_count == 2:
                    second_param = param_names[1]
                    fig = namespace['generate_chart'](historical_data, namespace.get(second_param, {}))
                else:
                    # Just use historical data as fallback
                    print("Using fallback with just historical data")
                    fig = namespace['generate_chart'](historical_data)
                
        return fig
            
    except Exception as e:
        print(f"Error executing chart script: {e}")
        import traceback
        traceback.print_exc()
        
        # Generate a simple fallback chart
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Close Price'
            ))
            fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")
            return fig
        except:
            return None
  

def create_comparative_analysis_prompt(tickers, all_stock_data, all_catalysts, company_names, user_prompt):
    """Create a prompt for comparative analysis of multiple stocks."""
    prompt = "You are a financial analyst. Below is the Yahoo Finance data for multiple stocks and their recent catalysts:\n\n"
    
    # Add stock data for each ticker
    for ticker in tickers:
        company_name = company_names[ticker]
        prompt += f"### Stock Data for {company_name} ({ticker}):\n{all_stock_data[ticker]}\n\n"
        
        # Add catalysts for each company
        if company_name in all_catalysts:
            prompt += f"### Recent Catalysts for {company_name}:\n{all_catalysts[company_name]}\n\n"
    
    # Add the user's question
    prompt += f"Based on all this information, provide a comparative analysis answering the following question. Include specific metrics, performance trends, and insights about relative strengths and weaknesses:\n{user_prompt}\n"
    
    return prompt

def generate_comparative_chart_script(tickers, all_stock_data, all_historical_data, user_prompt):
    """Generate a chart script for comparing multiple stocks."""
    # Prepare a sample of each ticker's historical data for the prompt
    historical_samples = {}
    for ticker in tickers:
        if not all_historical_data[ticker].empty:
            historical_samples[ticker] = all_historical_data[ticker].head().to_string()
    
    # Create enhanced prompt for OpenAI to generate a visualization script for multiple stocks
    chart_prompt = f"""
    Create a Python function that generates a comparative visualization specifically addressing this user question:
    "{user_prompt}"
    
    Tickers being compared: {', '.join(tickers)}
    
    Stock Summary Data:
    {', '.join([f"{ticker}: {all_stock_data[ticker].split('Long Business Summary')[0]}" for ticker in tickers])}
    
    The historical data for each ticker is available as separate DataFrames in a dictionary called 'historical_data_dict'.
    Example structure for '{tickers[0]}':
    {historical_samples.get(tickers[0], 'No data available')}
    
    Write a function called 'generate_chart' that:
    1. Takes historical_data_dict as the first parameter (a dictionary mapping ticker symbols to their historical DataFrame)
    2. Creates a comparative visualization using Plotly (import plotly.graph_objects as go)
    3. Shows metrics and trends SPECIFICALLY RELEVANT to comparing these stocks based on the user's question
    4. Handles cases where data might be missing for some tickers
    5. Uses clear labels, legends, and potentially different colors for each ticker
    6. Returns a plotly.graph_objects.Figure object
    
    For comparative charts, consider:
    - Normalized price charts (setting all stocks to 100% at a specific start date)
    - Side-by-side metrics comparisons
    - Performance metrics over the same time period
    
    Return ONLY the Python code without any explanation.
    """
    
    # Use the same LLM we already have
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    chart_prompt_template = PromptTemplate(
        input_variables=["chart_prompt"],
        template="{chart_prompt}"
    )
    
    chart_chain = LLMChain(llm=llm, prompt=chart_prompt_template)
    
    # Get the chart generation script
    try:
        chart_script = chart_chain.run(chart_prompt=chart_prompt).strip()
        print("=== Generated Comparative Chart Script ===")
        print(chart_script)
        print("=== End Comparative Chart Script ===\n")
        return chart_script
    except Exception as e:
        print(f"Error generating comparative chart script: {e}")
        return None

def execute_comparative_chart_script(script, all_historical_data, tickers):
    """Execute the generated comparative chart script and return a Plotly figure."""
    try:
        # Create a namespace for execution that includes Plotly and core libraries
        namespace = {
            'pd': pd,
            'plt': plt,
            'np': __import__('numpy'),
            'go': go,
            'make_subplots': make_subplots,
            'plotly': __import__('plotly'),
            'historical_data_dict': {},
            'dt': dt,
            'datetime': dt,
        }
        
        # Add historical data for each ticker
        for ticker in tickers:
            # Make a copy to avoid modifying the original
            historical_data = all_historical_data[ticker].copy() if ticker in all_historical_data else pd.DataFrame()
            
            # Normalize the datetime index by removing timezone information
            if not historical_data.empty and hasattr(historical_data.index, 'tz') and historical_data.index.tz is not None:
                historical_data.index = historical_data.index.tz_localize(None)
                
            namespace['historical_data_dict'][ticker] = historical_data
        
        # Execute the script in the namespace
        exec(script, namespace)
        
        # Check if generate_chart function was created
        if 'generate_chart' not in namespace:
            print("Error: generate_chart function not found in script")
            return create_fallback_comparative_chart(all_historical_data, tickers)

        # Call the generate_chart function
        fig = namespace['generate_chart'](namespace['historical_data_dict'])
        return fig
            
    except Exception as e:
        print(f"Error executing comparative chart script: {e}")
        import traceback
        traceback.print_exc()
        
        return create_fallback_comparative_chart(all_historical_data, tickers)

def create_fallback_comparative_chart(all_historical_data, tickers):
    """Create a simple fallback chart when the generated script fails."""
    try:
        fig = go.Figure()
        
        # Add each ticker to the chart
        for ticker in tickers:
            if ticker in all_historical_data and not all_historical_data[ticker].empty:
                fig.add_trace(go.Scatter(
                    x=all_historical_data[ticker].index,
                    y=all_historical_data[ticker]['Close'],
                    mode='lines',
                    name=f'{ticker} Close Price'
                ))
        
        fig.update_layout(
            title="Stock Price Comparison",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Tickers"
        )
        
        return fig
    except:
        return None


# Example usage:
#user_prompt = "Can you provide an analysis of Merck's recent performance?"
#print(answer_user_query(user_prompt))
#result = answer_user_query(user_prompt)
#print(result["text"])

