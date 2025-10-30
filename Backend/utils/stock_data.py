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

    def _get_exchange_from_symbol(self, symbol: str) -> str:
        """Determine the exchange from the symbol suffix."""
        if symbol.endswith('.NS'):
            return 'NSE (National Stock Exchange, India)'
        elif symbol.endswith('.BO'):
            return 'BSE (Bombay Stock Exchange, India)'
        else:
            return 'International Exchange'
    
    def _get_currency_symbol(self, symbol: str) -> str:
        """Get the appropriate currency symbol based on exchange."""
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            return '₹'  # Indian Rupee
        else:
            return '$'  # US Dollar (default)
    
    def _get_currency_code(self, symbol: str, info: dict) -> str:
        """
        Get the actual currency code from stock info.
        This is the authoritative source from yfinance data.
        """
        # First priority: Get actual currency from yfinance
        actual_currency = info.get('currency', info.get('financialCurrency', None))
        
        if actual_currency:
            return actual_currency.upper()
        
        # Fallback: Determine from symbol suffix
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            return 'INR'
        else:
            return 'USD'

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch comprehensive stock information for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS', 'TCS.BO')

        Returns:
            Dictionary containing stock information or None if error
        """
        try:
            self.logger.info(f"Fetching stock info for: {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Determine exchange and currency
            exchange = self._get_exchange_from_symbol(symbol)
            currency_symbol = self._get_currency_symbol(symbol)
            currency_code = self._get_currency_code(symbol, info)

            # Extract relevant information
            stock_data = {
                "symbol": symbol.upper(),
                "name": info.get("longName", "N/A"),
                "exchange": exchange,
                "currency": currency_symbol,
                "currency_code": currency_code,
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

            self.logger.info(f"Successfully fetched data for {symbol} from {exchange}")
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

        # Get currency symbol for formatting
        currency = stock_info.get("currency", "$")
        currency_code = stock_info.get("currency_code", "USD")
        exchange = stock_info.get("exchange", "N/A")
        
        # Determine if this is Indian stock
        is_indian_stock = symbol.endswith('.NS') or symbol.endswith('.BO')
        
        # Format market cap
        market_cap = stock_info["market_cap"]
        if isinstance(market_cap, (int, float)):
            market_cap_str = f"{currency}{market_cap:,.0f}"
        else:
            market_cap_str = str(market_cap)

        # Format basic stock info with clear currency indication
        context = f"""
REAL-TIME STOCK DATA FOR {stock_info["symbol"]} ({stock_info["name"]}):
Exchange: {exchange}
Currency: {currency_code} ({currency})
"""
        
        # Add important warning for Indian stocks
        if is_indian_stock:
            context += """
⚠️ IMPORTANT: All prices shown are in INDIAN RUPEES (INR/₹)
   - Do NOT convert these prices to INR (they are already in INR)
   - Do NOT apply USD to INR conversion rates
   - These are the actual trading prices on NSE/BSE in ₹
   
"""
        
        context += f"""Current Market Data (as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}):
- Current Price: {currency}{stock_info["current_price"]}
- Previous Close: {currency}{stock_info["previous_close"]}
- Day's Range: {currency}{stock_info["day_low"]} - {currency}{stock_info["day_high"]}
- Open Price: {currency}{stock_info["open"]}
- Volume: {stock_info["volume"]:,} shares
- Market Cap: {market_cap_str}

Financial Metrics:
- P/E Ratio: {stock_info["pe_ratio"]}
- EPS: {currency}{stock_info["eps"]}
- Dividend Yield: {stock_info["dividend_yield"]}

52-Week Performance:
- 52-Week High: {currency}{stock_info["52_week_high"]}
- 52-Week Low: {currency}{stock_info["52_week_low"]}

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
- Price Change: {currency}{hist_data["price_change"]:.2f} ({hist_data["price_change_percent"]:.2f}%)
- Average Close Price: {currency}{hist_data["average_close"]:.2f}
- Highest Price: {currency}{hist_data["highest_price"]:.2f}
- Lowest Price: {currency}{hist_data["lowest_price"]:.2f}
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
    
    def get_base_symbol(self, symbol: str) -> str:
        """
        Extract the base symbol without exchange suffix.
        
        Args:
            symbol: Full symbol (e.g., 'RELIANCE.NS', 'TCS.BO')
        
        Returns:
            Base symbol without suffix (e.g., 'RELIANCE', 'TCS')
        """
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            return symbol[:-3]
        return symbol
    
    def compare_nse_bse_prices(self, base_symbol: str) -> Optional[Dict]:
        """
        Compare prices of the same stock on NSE and BSE exchanges.
        Useful for Indian stocks to show price differences between exchanges.
        
        Args:
            base_symbol: Base stock symbol without exchange suffix (e.g., 'RELIANCE', 'TCS')
        
        Returns:
            Dictionary with NSE and BSE data comparison, or None if error
        """
        try:
            nse_symbol = f"{base_symbol}.NS"
            bse_symbol = f"{base_symbol}.BO"
            
            self.logger.info(f"Comparing NSE and BSE prices for {base_symbol}")
            
            nse_data = self.get_stock_info(nse_symbol)
            bse_data = self.get_stock_info(bse_symbol)
            
            if not nse_data and not bse_data:
                self.logger.warning(f"Could not fetch data from either exchange for {base_symbol}")
                return None
            
            comparison = {
                'base_symbol': base_symbol,
                'company_name': nse_data.get('name') if nse_data else bse_data.get('name'),
                'nse': nse_data,
                'bse': bse_data,
                'price_difference': None,
                'price_difference_percent': None,
                'recommendation': 'NSE (National Stock Exchange) - More liquid and preferred'
            }
            
            # Calculate price difference if both are available
            if nse_data and bse_data:
                nse_price = nse_data.get('current_price')
                bse_price = bse_data.get('current_price')
                
                if isinstance(nse_price, (int, float)) and isinstance(bse_price, (int, float)):
                    price_diff = nse_price - bse_price
                    price_diff_percent = (price_diff / bse_price) * 100
                    
                    comparison['price_difference'] = price_diff
                    comparison['price_difference_percent'] = price_diff_percent
                    
                    # Add note about price difference
                    if abs(price_diff_percent) > 0.1:
                        comparison['note'] = f"Price difference: ₹{abs(price_diff):.2f} ({abs(price_diff_percent):.2f}%)"
                    else:
                        comparison['note'] = "Prices are approximately the same on both exchanges"
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing NSE/BSE prices for {base_symbol}: {str(e)}")
            return None
    
    def format_nse_bse_comparison(self, base_symbol: str) -> str:
        """
        Format NSE vs BSE comparison into readable text for LLM context.
        
        Args:
            base_symbol: Base stock symbol without exchange suffix
        
        Returns:
            Formatted comparison text
        """
        comparison = self.compare_nse_bse_prices(base_symbol)
        
        if not comparison:
            return f"Unable to compare NSE/BSE prices for {base_symbol}"
        
        context = f"""
NSE vs BSE PRICE COMPARISON FOR {comparison['base_symbol']} ({comparison['company_name']}):
"""
        
        if comparison['nse']:
            nse_price = comparison['nse'].get('current_price', 'N/A')
            nse_volume = comparison['nse'].get('volume', 'N/A')
            context += f"""
NSE (National Stock Exchange):
- Current Price: ₹{nse_price}
- Volume: {nse_volume:,} shares
- Exchange: Preferred for higher liquidity
"""
        
        if comparison['bse']:
            bse_price = comparison['bse'].get('current_price', 'N/A')
            bse_volume = comparison['bse'].get('volume', 'N/A')
            context += f"""
BSE (Bombay Stock Exchange):
- Current Price: ₹{bse_price}
- Volume: {bse_volume:,} shares
- Exchange: Alternative exchange
"""
        
        if comparison.get('note'):
            context += f"\n{comparison['note']}\n"
        
        context += """
IMPORTANT NOTE: Indian stocks trade on both NSE and BSE with independent pricing.
- NSE generally has higher liquidity and trading volume
- Small price differences are normal due to supply/demand on each exchange
- For investing, NSE (.NS) is typically preferred due to better liquidity
- Both exchanges are regulated by SEBI and equally safe
"""
        
        return context.strip()
