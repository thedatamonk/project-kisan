import os
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from mods.tools.tool_schema import tool_schema
from mods.tools.tool_types import ToolResponse

load_dotenv()


class AgroMarketAnalyserTool:
    """
    LLM-compatible tool for fetching and analyzing Indian agricultural market data.
    Provides structured responses optimized for agent interpretation.
    """
    def __init__(self):
        """Initialize the mandi market data tool"""
        self.base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.api_key = os.getenv("DATA_GOV_IN_API_KEY")
        
        if not self.api_key:
            raise ValueError("DATA_GOV_IN_API_KEY not found in environment variables")
    
    @tool_schema(
        description=(
            "Fetches live market prices and data from Indian agricultural mandis (wholesale markets). "
            "Use this tool when users ask about: "
            "1) Current prices of agricultural commodities (vegetables, fruits, grains) "
            "2) Price comparisons across different markets or regions "
            "3) Market availability of specific crops "
            "4) Price trends in specific states, districts, or markets. "
            "The tool returns recent market data including commodity names, prices (min/max/modal), "
            "markets, arrival quantities, and dates."
        ),
        commodity_description=(
            "Name of the agricultural commodity to query. "
            "Examples: 'Tomato', 'Potato', 'Onion', 'Rice', 'Wheat', 'Brinjal'. "
            "Use title case. Leave None to get data for all commodities."
        ),
        state_description=(
            "Indian state name to filter results. "
            "Examples: 'Karnataka', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'Uttar Pradesh'. "
            "Use title case. Leave None to search across all states."
        ),
        district_description=(
            "District name within the state to filter results. "
            "Examples: 'Bangalore', 'Pune', 'Ludhiana', 'Chennai'. "
            "Use title case. Leave None to search across all districts."
        ),
        market_description=(
            "Specific market/mandi name to filter results. "
            "Examples: 'Bangalore', 'Azadpur', 'Vashi'. "
            "Use title case. Leave None to search across all markets."
        ),
        limit_description=(
            "Maximum number of market records to fetch. "
            "Valid range: 1-100. Default is 10. "
            "Use higher values (20-50) for price comparisons across multiple markets."
        )
    )
    def execute(
        self,
        commodity: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 10
    ) -> ToolResponse:
        """
        Main execution method called by the LLM agent.
        
        Args:
            commodity: Commodity name (e.g., "Tomato")
            state: State name (e.g., "Karnataka")
            district: District name (e.g., "Bangalore")
            market: Market name (e.g., "Bangalore")
            limit: Number of records (1-100)
            
        Returns:
            Structured response with success status, data, and human-readable summary
        """

        # Validate limit
        limit = max(1, min(limit, 100))
        
        # Fetch raw data from API
        raw_response = self._fetch_api_data(
            commodity=commodity,
            state=state,
            district=district,
            market=market,
            limit=limit
        )

        # Handle API errors
        if "error" in raw_response:
            return ToolResponse(
                success=False,
                data=None,
                error=raw_response["error"],
                message="Failed to fetch market data from API. Please check your internet connection or try again later."
            )
        
        # Process and structure the response
        processed_data = self._process_response(raw_response)
        
        # Generate human-readable summary for the LLM
        summary = self._generate_summary(processed_data, commodity, state, district, market)

        # Build metadata for context
        metadata = {
            "query_params": {
                "commodity": commodity,
                "state": state,
                "district": district,
                "market": market,
                "limit": limit
            },
            "record_count": len(processed_data.get("records", [])),
            "statistics": processed_data.get("metadata", {})
        }

        return ToolResponse(
            success=True,
            data={
                "records": processed_data.get("records", []),
                "metadata": metadata
            },
            message=summary
        )
    
    def _fetch_api_data(
        self,
        commodity: Optional[str],
        state: Optional[str],
        district: Optional[str],
        market: Optional[str],
        limit: int
    ) -> dict:
        """Fetch raw data from the data.gov.in API"""
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": 0
        }
        
        # Add filters
        if commodity:
            params["filters[commodity]"] = commodity
        if state:
            params["filters[state]"] = state
        if district:
            params["filters[district]"] = district
        if market:
            params["filters[market]"] = market
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "API request timed out", "status": "failed"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}

    def _process_response(self, raw_response: dict) -> ToolResponse:
        """
        Process raw API response into structured format.
        Handles missing fields and data cleaning.
        """
        if "records" not in raw_response:
            return {"records": [], "metadata": {}}
        
        records = raw_response.get("records", [])
        processed_records = []
        
        for record in records:
            # Extract and clean fields
            processed_record = {
                "commodity": record.get("commodity", "Unknown"),
                "state": record.get("state", "Unknown"),
                "district": record.get("district", "Unknown"),
                "market": record.get("market", "Unknown"),
                "min_price": self._safe_float(record.get("min_price")),
                "max_price": self._safe_float(record.get("max_price")),
                "modal_price": self._safe_float(record.get("modal_price")),
                "price_date": record.get("price_date", "Unknown"),
                "arrival_date": record.get("arrival_date", "Unknown"),
                "variety": record.get("variety", "Not specified"),
                "grade": record.get("grade", "Not specified")
            }
            processed_records.append(processed_record)
        
        # Calculate aggregate statistics
        metadata = self._calculate_statistics(processed_records)
        
        return {
            "records": processed_records,
            "metadata": metadata
        }
    
    def _calculate_statistics(self, records: list[dict]) -> ToolResponse:
        """Calculate aggregate statistics from records"""
        if not records:
            return {}
        
        modal_prices = [r["modal_price"] for r in records if r["modal_price"] is not None]
        min_prices = [r["min_price"] for r in records if r["min_price"] is not None]
        max_prices = [r["max_price"] for r in records if r["max_price"] is not None]

        stats = {
            "total_records": len(records),
            "unique_markets": len(set(r["market"] for r in records)),
            "unique_states": len(set(r["state"] for r in records)),
            "unique_districts": len(set(r["district"] for r in records)),
        }
        
        if modal_prices:
            stats.update({
                "avg_modal_price": round(sum(modal_prices) / len(modal_prices), 2),
                "min_modal_price": min(modal_prices),
                "max_modal_price": max(modal_prices)
            })
        
        if min_prices:
            stats["avg_min_price"] = round(sum(min_prices) / len(min_prices), 2)
        
        if max_prices:
            stats["avg_max_price"] = round(sum(max_prices) / len(max_prices), 2)
        
        return stats

    def _generate_summary(
        self,
        processed_data: dict,
        commodity: Optional[str],
        state: Optional[str],
        district: Optional[str],
        market: Optional[str]
    ) -> str:
        """Generate human-readable summary for LLM interpretation"""

        records = processed_data.get("records", [])
        metadata = processed_data.get("metadata", {})
        
        if not records:
            filters = []
            if commodity:
                filters.append(f"commodity '{commodity}'")
            if state:
                filters.append(f"state '{state}'")
            if district:
                filters.append(f"district '{district}'")
            if market:
                filters.append(f"market '{market}'")
            
            filter_text = " and ".join(filters) if filters else "the specified criteria"
            return f"No market data found for {filter_text}. Try broadening your search or checking the spelling."
        
        # Build summary
        summary_parts = []
        summary_parts.append(f"Found {len(records)} market record(s)")
        
        if commodity:
            summary_parts.append(f"for {commodity}")
        
        if metadata.get("avg_modal_price"):
            avg_price = metadata["avg_modal_price"]
            min_price = metadata["min_modal_price"]
            max_price = metadata["max_modal_price"]
            summary_parts.append(
                f"with average modal price of ₹{avg_price}/quintal "
                f"(range: ₹{min_price} - ₹{max_price})"
            )
        
        summary_parts.append(f"across {metadata.get('unique_markets', 0)} market(s)")
        
        if state:
            summary_parts.append(f"in {state}")
        elif metadata.get("unique_states", 0) > 1:
            summary_parts.append(f"across {metadata['unique_states']} states")
        
        return " ".join(summary_parts) + "."
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float, return None if conversion fails"""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
        
if __name__ == "__main__":
    # Example 1: Query tomato prices in Karnataka
    tool = AgroMarketAnalyserTool()

    print("Example 1 - Tomato prices in Karnataka:")
    result = tool.execute(commodity="Tomato", state="Karnataka", limit=5)
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Records found: {result['data']['metadata']['record_count']}")

    if result['success'] and result['data']['records']:
        print("\nSample record:")
        sample = result['data']['records'][0]
        print(f"  Market: {sample['market']}")
        print(f"  Modal Price: ₹{sample['modal_price']}/quintal")
        print(f"  Date: {sample['price_date']}")
    
    print("\n" + "="*60 + "\n")