from __future__ import annotations

import colorsys


_CLASS_PALETTE = (
    "#e53935",
    "#1e88e5",
    "#43a047",
    "#fb8c00",
    "#8e24aa",
    "#00acc1",
    "#fdd835",
    "#6d4c41",
    "#d81b60",
    "#3949ab",
    "#7cb342",
    "#f4511e",
)


def class_color(class_id: int) -> str:
    """Return a stable, visually distinct color for a non-negative class ID."""
    class_id = int(class_id)
    if class_id < 0:
        return "#9e9e9e"
    if class_id < len(_CLASS_PALETTE):
        return _CLASS_PALETTE[class_id]
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def class_display_name(class_id: int, name: str) -> str:
    return f"{int(class_id)} — {name}"
