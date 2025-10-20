import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class AgroMarketAnalyserTool:
    """
    Client for accessing Indian agricultural market data via data.gov.in API
    """
    
    def __init__(self):
        """
        Initialize API client
        
        Args:
            api_key: API key from data.gov.in (default is public key with 10 record limit)
        """
        
        self.base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        # self.base_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
        self.api_key = os.getenv("DATA_GOV_IN_API_KEY")
    
    def get_commodity_prices(
        self,
        commodity: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        variety: Optional[str] = None,
        grade: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict:
        """
        Fetch commodity prices from mandis
        
        Args:
            commodity: Name of commodity (e.g., "Tomato", "Potato", "Onion")
            state: State name (e.g., "Karnataka", "Maharashtra")
            district: District name (e.g., "Bangalore")
            market: Market name (e.g., "Bangalore")
            limit: Number of records to fetch (max 10 for public API key)
            offset: Offset for pagination
            
        Returns:
            Dictionary containing API response with price data
        """

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": offset
        }
        
        # Add filters if provided
        if commodity:
            params["filters[commodity]"] = commodity
        if state:
            params["filters[state]"] = state
        if district:
            params["filters[district]"] = district
        if market:
            params["filters[market]"] = market
        if variety:
            params["filters[variety]"] = variety
        if grade:
            params["filters[grade]"] = grade
            
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
        