from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class OrderSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    type: OrderType
    size: float
    price: float | None = None
    stop_price: float | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    size: float
    price: float
    fee: float
    slippage: float
    ts_ms: int
