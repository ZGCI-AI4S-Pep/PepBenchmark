import matplotlib.colors as mcolors
import matplotlib.cm as cm
from typing import Tuple, List
import numpy as np


def rgb2hex(rgb: Tuple) -> str:
    rgb = tuple(rgb)
    # Normalize to [0,1] before converting
    hex_color = mcolors.to_hex([c / 255 for c in rgb])
    print(hex_color)  # -> #ff6432


def get_color_gradient(hex_color: str, bins: int = 10) -> List[str]:
    """
    Create a list of colors from white to the given hex color.
    """
    base_rgb = np.array(mcolors.to_rgb(hex_color))  # target color (0-1 scale)
    white = np.array([1, 1, 1])  # pure white
    gradients = [
        mcolors.to_hex(white * (1 - t) + base_rgb * t) for t in np.linspace(0, 1, bins)
    ]
    return gradients


def get_cool_warm_gradient() -> Tuple[List[str]]:
    # Get colormap
    cmap = cm.get_cmap("coolwarm")

    # 10 bins from blue→white (left half)
    blue_side = [cmap(i / 20) for i in range(0, 10)]

    # 10 bins from white→red (right half)
    red_side = [cmap(i / 20) for i in range(10, 20)]

    # Convert to hex for easy use
    blues = [mcolors.to_hex(c) for c in blue_side]
    reds = [mcolors.to_hex(c) for c in red_side]

    blues = list(reversed(blues))  # from shallow to deep
    return blues, reds
