import backtrader as bt
import pandas as pd
import yfinance as yf
import sqlite3
import boto3
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import json
import os
import hashlib
from functools import lru_cache
import redis
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DataValidator:
    """Validates data quality and structure"""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> DataValidationResult:
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            
        # Check for missing values
        if df.isnull().any().any():
            warnings.append("Data contains missing values")
            
        # Check for price anomalies
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            if (price_changes > 0.5).any():  # 50% price change
                warnings.append("Large price changes detected")
                
        # Check for volume anomalies
        if 'volume' in df.columns:
            volume_mean = df['volume'].mean()
            if (df['volume'] > volume_mean * 10).any():
                warnings.append("Unusual volume spikes detected")
                
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

class CacheManager:
    """Manages data caching using Redis or local LRU cache"""
    
    def __init__(self, use_redis: bool = False, redis_url: str = None):
        self.use_redis = use_redis
        if use_redis:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.local_cache = {}
            
    def get(self, key: str) -> Optional[Any]:
        if self.use_redis:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        return self.local_cache.get(key)
        
    def set(self, key: str, value: Any, expiry: int = 3600):
        if self.use_redis:
            self.redis_client.setex(key, expiry, json.dumps(value))
        else:
            self.local_cache[key] = value
            
    def delete(self, key: str):
        if self.use_redis:
            self.redis_client.delete(key)
        else:
            self.local_cache.pop(key, None)

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager
        self.validator = DataValidator()
        
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, data_type: str) -> str:
        """Generate a unique cache key"""
        key_str = f"{symbol}_{start_date}_{end_date}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data for a symbol between start_date and end_date"""
        pass
    
    @abstractmethod
    def get_metrics(self, symbol: str, start_date: str, end_date: str, metrics: List[str]) -> Dict:
        """Get specific metrics for a symbol between start_date and end_date"""
        pass
        
    def validate_data(self, df: pd.DataFrame) -> DataValidationResult:
        """Validate the data quality"""
        return self.validator.validate_price_data(df)

class YFinanceDataSource(DataSource):
    """Yahoo Finance data source implementation"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, max_workers: int = 5):
        super().__init__(cache_manager)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_key = self._get_cache_key(symbol, start_date, end_date, 'price_data')
        
        # Try to get from cache
        if self.cache_manager:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                return pd.DataFrame(cached_data)
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # Add dividends as a separate column
            dividends = ticker.dividends
            if not dividends.empty:
                df['dividends'] = dividends
            else:
                df['dividends'] = 0.0
                
            # Validate data
            validation = self.validate_data(df)
            if not validation.is_valid:
                logger.error(f"Data validation failed: {validation.errors}")
                raise ValueError("Data validation failed")
            if validation.warnings:
                logger.warning(f"Data validation warnings: {validation.warnings}")
                
            # Cache the data
            if self.cache_manager:
                self.cache_manager.set(cache_key, df.to_dict())
                
            return df
        except Exception as e:
            logger.error(f"Error fetching data from YFinance for {symbol}: {str(e)}")
            raise
            
    def get_metrics(self, symbol: str, start_date: str, end_date: str, metrics: List[str]) -> Dict:
        cache_key = self._get_cache_key(symbol, start_date, end_date, f"metrics_{','.join(metrics)}")
        
        # Try to get from cache
        if self.cache_manager:
            cached_metrics = self.cache_manager.get(cache_key)
            if cached_metrics is not None:
                return cached_metrics
        
        try:
            ticker = yf.Ticker(symbol)
            result = {}
            
            # Define metric getters
            metric_getters = {
                'pe_ratio': lambda: ticker.info.get('trailingPE'),
                'market_cap': lambda: ticker.info.get('marketCap'),
                'dividend_yield': lambda: ticker.info.get('dividendYield'),
                'beta': lambda: ticker.info.get('beta'),
                'volatility': lambda: ticker.info.get('regularMarketPrice') * ticker.info.get('regularMarketVolume', 0),
                'sector': lambda: ticker.info.get('sector'),
                'industry': lambda: ticker.info.get('industry'),
                'eps': lambda: ticker.info.get('trailingEps'),
                'book_value': lambda: ticker.info.get('bookValue'),
                'price_to_book': lambda: ticker.info.get('priceToBook')
            }
            
            # Fetch metrics in parallel
            futures = []
            for metric in metrics:
                if metric in metric_getters:
                    futures.append(
                        self.executor.submit(metric_getters[metric])
                    )
            
            # Collect results
            for metric, future in zip(metrics, futures):
                result[metric] = future.result()
                
            # Cache the results
            if self.cache_manager:
                self.cache_manager.set(cache_key, result)
                
            return result
        except Exception as e:
            logger.error(f"Error fetching metrics from YFinance for {symbol}: {str(e)}")
            raise

class SQLiteDataSource(DataSource):
    """SQLite data source implementation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT date, open, high, low, close, volume, dividends
            FROM price_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data from SQLite for {symbol}: {str(e)}")
            raise
        finally:
            conn.close()
            
    def get_metrics(self, symbol: str, start_date: str, end_date: str, metrics: List[str]) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            result = {}
            
            for metric in metrics:
                query = f"""
                SELECT value FROM metrics
                WHERE symbol = ? AND metric = ? AND date BETWEEN ? AND ?
                ORDER BY date DESC LIMIT 1
                """
                cursor = conn.execute(query, (symbol, metric, start_date, end_date))
                value = cursor.fetchone()
                result[metric] = value[0] if value else None
                
            return result
        except Exception as e:
            logger.error(f"Error fetching metrics from SQLite for {symbol}: {str(e)}")
            raise
        finally:
            conn.close()

class S3DataSource(DataSource):
    """AWS S3 data source implementation"""
    
    def __init__(self, bucket_name: str, region_name: str = 'us-east-1'):
        self.s3 = boto3.client('s3', region_name=region_name)
        self.bucket_name = bucket_name
        
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            key = f"data/{symbol}/price_data.csv"
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            df = pd.read_csv(response['Body'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Filter by date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df[mask]
        except Exception as e:
            logger.error(f"Error fetching data from S3 for {symbol}: {str(e)}")
            raise
            
    def get_metrics(self, symbol: str, start_date: str, end_date: str, metrics: List[str]) -> Dict:
        try:
            result = {}
            for metric in metrics:
                key = f"metrics/{symbol}/{metric}.json"
                response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
                metric_data = json.loads(response['Body'].read().decode('utf-8'))
                
                # Filter by date range
                filtered_data = {
                    k: v for k, v in metric_data.items()
                    if start_date <= k <= end_date
                }
                result[metric] = filtered_data
                
            return result
        except Exception as e:
            logger.error(f"Error fetching metrics from S3 for {symbol}: {str(e)}")
            raise

class BacktraderDataProvider(bt.feeds.PandasData):
    """Custom Backtrader data provider that supports multiple data sources"""
    
    params = (
        ('data_source', None),  # DataSource instance
        ('symbol', None),       # Trading symbol
        ('start_date', None),   # Start date
        ('end_date', None),     # End date
        ('dividends', 'dividends'),  # Column name for dividends
        ('validate_data', True),  # Whether to validate data
    )
    
    def __init__(self):
        super().__init__()
        self.data_source = self.p.data_source
        self.symbol = self.p.symbol
        self.start_date = self.p.start_date
        self.end_date = self.p.end_date
        
        # Fetch data from the data source
        df = self.data_source.get_data(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Validate data if required
        if self.p.validate_data:
            validation = self.data_source.validate_data(df)
            if not validation.is_valid:
                raise ValueError(f"Data validation failed: {validation.errors}")
            if validation.warnings:
                logger.warning(f"Data validation warnings: {validation.warnings}")
        
        # Set the data
        self.data = df
        
    def get_metrics(self, metrics: List[str]) -> Dict:
        """Get specific metrics for the symbol"""
        return self.data_source.get_metrics(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            metrics=metrics
        )

# Example usage with enhanced features:
"""
# Initialize cache manager
cache_manager = CacheManager(use_redis=True, redis_url='redis://localhost:6379')

# Using YFinance with caching and validation
yf_source = YFinanceDataSource(cache_manager=cache_manager)
data = BacktraderDataProvider(
    data_source=yf_source,
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    validate_data=True
)

# Get metrics with caching
metrics = data.get_metrics(['pe_ratio', 'market_cap', 'dividend_yield', 'beta', 'volatility'])
""" 