"""
A tiny in-memory mock order DB, standing in for a real order-management-system API.
Real tool-calling agents in production call an internal service here instead -- the
function signatures (get_order_status, check_return_eligibility) are written so swapping
this module for a real HTTP client is a one-file change; app/tool_agent.py never touches
this data directly, it only calls these two functions.
"""
from __future__ import annotations

import datetime as dt

_TODAY = dt.date(2026, 10, 18)  # fixed "today" so the demo is reproducible; see decisions_log.md

_ORDERS = {
    "NV-100234": {
        "status": "Shipped",
        "items": ["NovaVolt NV100 Wireless Earbuds"],
        "ordered_on": dt.date(2026, 10, 15),
        "delivered_on": None,
        "estimated_delivery": dt.date(2026, 10, 20),
        "amount_paid": 2499,
    },
    "NV-100199": {
        "status": "Delivered",
        "items": ["NovaVolt PD65 USB-C Laptop Charger"],
        "ordered_on": dt.date(2026, 10, 2),
        "delivered_on": dt.date(2026, 10, 6),
        "estimated_delivery": dt.date(2026, 10, 6),
        "amount_paid": 1799,
    },
    "NV-100087": {
        "status": "Delivered",
        "items": ["NovaVolt BS5 Portable Bluetooth Speaker"],
        "ordered_on": dt.date(2026, 9, 10),
        "delivered_on": dt.date(2026, 9, 14),
        "estimated_delivery": dt.date(2026, 9, 14),
        "amount_paid": 3299,
    },
    "NV-100042": {
        "status": "Out for Delivery",
        "items": ["NovaVolt SP2 Smart Wi-Fi Plug", "NovaVolt PB20000 Power Bank"],
        "ordered_on": dt.date(2026, 10, 16),
        "delivered_on": None,
        "estimated_delivery": dt.date(2026, 10, 18),
        "amount_paid": 899 + 2199,
    },
}


def get_order_status(order_id: str) -> dict:
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)
    if not order:
        return {"found": False, "order_id": order_id, "error": "No order found with this ID."}
    return {
        "found": True,
        "order_id": order_id,
        "status": order["status"],
        "items": order["items"],
        "ordered_on": order["ordered_on"].isoformat(),
        "estimated_delivery": order["estimated_delivery"].isoformat(),
        "delivered_on": order["delivered_on"].isoformat() if order["delivered_on"] else None,
    }


def check_return_eligibility(order_id: str) -> dict:
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)
    if not order:
        return {"found": False, "order_id": order_id, "error": "No order found with this ID."}

    if order["status"] != "Delivered" or not order["delivered_on"]:
        return {
            "found": True,
            "order_id": order_id,
            "eligible": False,
            "reason": "Order has not been delivered yet, so the return window hasn't started.",
        }

    days_since_delivery = (_TODAY - order["delivered_on"]).days
    eligible = days_since_delivery <= 10
    return {
        "found": True,
        "order_id": order_id,
        "eligible": eligible,
        "days_since_delivery": days_since_delivery,
        "window_days": 10,
        "amount_paid": order["amount_paid"],
        "reason": (
            "Within the 10-day return window." if eligible
            else f"Delivered {days_since_delivery} days ago, which is past the 10-day return window."
        ),
    }
