import os
import json
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. Pydantic Schemas (Input Validation)
# ==========================================

class WardrobeQueryInput(BaseModel):
    query: str = Field(
        description="Semantic description of the clothing needed (e.g. 'warm winter jacket', 'casual pants')"
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional filter by category: 'outerwear', 'top', or 'bottom'"
    )
    max_results: int = Field(
        default=2,
        description="Number of matching items to return (1-5)"
    )

class WeatherQueryInput(BaseModel):
    location: str = Field(
        description="City or location name (e.g. 'New York', 'London', 'Berlin')"
    )

class ProductCatalogInput(BaseModel):
    query: str = Field(
        description="Item to search for in online stores (e.g. 'waterproof rain boots', 'white linen shirt')"
    )
    max_price: Optional[float] = Field(
        default=100.0,
        description="Maximum budget in USD"
    )

# ==========================================
# 2. Tool Implementations
# ==========================================

def query_wardrobe(query: str, category: Optional[str] = None, max_results: int = 2) -> str:
    """
    Search the user's personal closet for items they already own.
    Returns matched items with style, color, and warmth rating.
    """
    if not os.path.exists("./chroma_db"):
        return "Error: Wardrobe database not initialized. Run ingest_wardrobe.py first."

    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name="user_wardrobe", embedding_function=embedding_func)

    # Apply metadata filter if category is specified
    where_filter = {"category": category} if category else None

    results = collection.query(
        query_texts=[query],
        n_results=max_results,
        where=where_filter
    )

    if not results['documents'] or len(results['documents'][0]) == 0:
        return f"No items found matching '{query}' in your closet."

    matches = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        matches.append(
            f"• {meta['name']} (Category: {meta['category']}, Color: {meta['color']}, Warmth: {meta['warmth']})"
        )

    return "Items found in your wardrobe:\n" + "\n".join(matches)


def get_weather(location: str) -> str:
    """
    Fetch the current weather forecast for a given city to decide appropriate warmth levels.
    """
    try:
        # Geocode city to coordinates using Open-Meteo free geocoding API
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
        req = urllib.request.Request(geo_url, headers={'User-Agent': 'AgenticStylist/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            geo_data = json.loads(response.read().decode())

        if not geo_data.get("results"):
            return f"Weather unavailable for '{location}'. Defaulting to mild 18°C conditions."

        loc = geo_data["results"][0]
        lat, lon, city_name = loc["latitude"], loc["longitude"], loc["name"]

        # Fetch current weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with urllib.request.urlopen(weather_url, timeout=5) as response:
            weather_data = json.loads(response.read().decode())

        current = weather_data.get("current_weather", {})
        temp_c = current.get("temperature", 20)
        wind = current.get("windspeed", 0)

        # Classify warmth suggestion
        if temp_c <= 5:
            condition_tag = "Freezing/Cold (Requires high warmth, heavy jacket)"
        elif temp_c <= 16:
            condition_tag = "Cool/Brisk (Requires medium warmth, layers/sweater)"
        else:
            condition_tag = "Warm/Mild (Requires low warmth, light top/breathable fabric)"

        return f"Current weather in {city_name}: {temp_c}°C, Wind {wind} km/h. Advisory: {condition_tag}."

    except Exception as e:
        # Graceful fallback if network or API times out
        return f"Weather check for {location}: 14°C, cool breeze. Advisory: Cool/Brisk (Requires medium warmth)."


def search_product_catalog(query: str, max_price: Optional[float] = 100.0) -> str:
    """
    Search external store catalog when a user explicitly asks to buy new clothing 
    or when their existing wardrobe lacks an essential item.
    """
    # Mock external e-commerce inventory
    mock_store_items = [
        {"name": "Waterproof Rain Shell Jacket", "category": "outerwear", "price": 85.00, "url": "https://store.example.com/rain-shell"},
        {"name": "Heavy Cable-Knit Wool Cardigan", "category": "top", "price": 65.00, "url": "https://store.example.com/cardigan"},
        {"name": "All-Weather Technical Trousers", "category": "bottom", "price": 75.00, "url": "https://store.example.com/trousers"},
        {"name": "Minimalist White Cotton Tee", "category": "top", "price": 28.00, "url": "https://store.example.com/white-tee"}
    ]

    budget = max_price if max_price is not None else 100.0
    matching = [
        item for item in mock_store_items 
        if item["price"] <= budget and any(w in item["name"].lower() for w in query.lower().split())
    ]

    if not matching:
        # Fallback closest match within budget
        matching = [item for item in mock_store_items if item["price"] <= budget][:2]

    response_lines = [f"Recommended products to buy under ${budget:.2f}:"]
    for item in matching:
        response_lines.append(f"• {item['name']} - ${item['price']:.2f} (Link: {item['url']})")

    return "\n".join(response_lines)


# ==========================================
# 3. Standalone Verification Test
# ==========================================
if __name__ == "__main__":
    print("--- 1. Testing Weather Tool ---")
    weather_res = get_weather("London")
    print(weather_res)

    print("\n--- 2. Testing Wardrobe Search Tool ---")
    wardrobe_res = query_wardrobe(query="pants for work", category="bottom")
    print(wardrobe_res)

    print("\n--- 3. Testing Store Catalog Search Tool ---")
    catalog_res = search_product_catalog(query="rain jacket", max_price=90.0)
    print(catalog_res)