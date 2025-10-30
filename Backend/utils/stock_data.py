import yfinance as yf
from typing import Dict, Optional, List
from datetime import datetime
from .logger import CustomLogger


class StockDataFetcher:
    """
    Fetches and formats stock market data using yfinance API.
    Provides real-time quotes, company info, and historical data.
    """

    def __init__(self, logger: CustomLogger):
        """
        Initialize the stock data fetcher.

        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch comprehensive stock information for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

        Returns:
            Dictionary containing stock information or None if error
        """
        try:
            self.logger.info(f"Fetching stock info for: {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant information
            stock_data = {
                "symbol": symbol.upper(),
                "name": info.get("longName", "N/A"),
                "current_price": info.get(
                    "currentPrice", info.get("regularMarketPrice", "N/A")
                ),
                "previous_close": info.get("previousClose", "N/A"),
                "open": info.get("open", info.get("regularMarketOpen", "N/A")),
                "day_high": info.get(
                    "dayHigh", info.get("regularMarketDayHigh", "N/A")
                ),
                "day_low": info.get("dayLow", info.get("regularMarketDayLow", "N/A")),
                "volume": info.get("volume", info.get("regularMarketVolume", "N/A")),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
            }

            self.logger.info(f"Successfully fetched data for {symbol}")
            return stock_data

        except Exception as e:
            self.logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return None

    def get_historical_data(self, symbol: str, period: str = "1mo") -> Optional[Dict]:
        """
        Fetch historical stock data.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Dictionary with historical data summary or None if error
        """
        try:
            self.logger.info(
                f"Fetching historical data for {symbol} (period: {period})"
            )
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                self.logger.info(f"No historical data found for {symbol}", "warning")
                return None

            # Calculate summary statistics
            historical_summary = {
                "symbol": symbol.upper(),
                "period": period,
                "start_date": hist.index[0].strftime("%Y-%m-%d"),
                "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                "highest_price": float(hist["High"].max()),
                "lowest_price": float(hist["Low"].min()),
                "average_close": float(hist["Close"].mean()),
                "total_volume": int(hist["Volume"].sum()),
                "price_change": float(hist["Close"].iloc[-1] - hist["Close"].iloc[0]),
                "price_change_percent": float(
                    (
                        (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                        / hist["Close"].iloc[0]
                    )
                    * 100
                ),
                "current_close": float(hist["Close"].iloc[-1]),
            }

            self.logger.info(f"Successfully fetched historical data for {symbol}")
            return historical_summary

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Fetch stock information for multiple symbols.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to their stock data
        """
        results = {}
        for symbol in symbols:
            results[symbol.upper()] = self.get_stock_info(symbol)
        return results

    def format_stock_context(self, symbol: str, include_historical: bool = True) -> str:
        """
        Format stock data into a context string for LLM.

        Args:
            symbol: Stock ticker symbol
            include_historical: Whether to include historical data (1 month)

        Returns:
            Formatted string with stock information
        """
        stock_info = self.get_stock_info(symbol)

        if not stock_info:
            return f"Unable to fetch data for symbol: {symbol}"

        # Format basic stock info
        context = f"""
REAL-TIME STOCK DATA FOR {stock_info["symbol"]} ({stock_info["name"]}):

Current Market Data (as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}):
- Current Price: ${stock_info["current_price"]}
- Previous Close: ${stock_info["previous_close"]}
- Day's Range: ${stock_info["day_low"]} - ${stock_info["day_high"]}
- Open Price: ${stock_info["open"]}
- Volume: {stock_info["volume"]:,} shares
- Market Cap: ${stock_info["market_cap"]:,} if isinstance(stock_info['market_cap'], (int, float)) else stock_info['market_cap']

Financial Metrics:
- P/E Ratio: {stock_info["pe_ratio"]}
- EPS: {stock_info["eps"]}
- Dividend Yield: {stock_info["dividend_yield"]}

52-Week Performance:
- 52-Week High: ${stock_info["52_week_high"]}
- 52-Week Low: ${stock_info["52_week_low"]}

Company Information:
- Sector: {stock_info["sector"]}
- Industry: {stock_info["industry"]}
"""

        # Add historical data if requested
        if include_historical:
            hist_data = self.get_historical_data(symbol, period="1mo")
            if hist_data:
                context += f"""
Historical Performance (Last Month):
- Period: {hist_data["start_date"]} to {hist_data["end_date"]}
- Price Change: ${hist_data["price_change"]:.2f} ({hist_data["price_change_percent"]:.2f}%)
- Average Close Price: ${hist_data["average_close"]:.2f}
- Highest Price: ${hist_data["highest_price"]:.2f}
- Lowest Price: ${hist_data["lowest_price"]:.2f}
- Total Trading Volume: {hist_data["total_volume"]:,} shares
"""

        return context.strip()

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists.

        Args:
            symbol: Stock ticker symbol to validate

        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Check if we got actual data back
            return "regularMarketPrice" in info or "currentPrice" in info
        except Exception:
            return False
