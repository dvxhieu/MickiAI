"""Example weather tool - demonstrates the @register_tool decorator."""

from app.tools.registry import register_tool

WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": "Lấy thông tin thời tiết hiện tại của một thành phố",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@register_tool("get_weather", WEATHER_SCHEMA)
def get_weather(city: str) -> str:
    # 实际会在真实的 API 调用这里
    return f"Thời tiết tại {city}: 30°C, nắng nhẹ"