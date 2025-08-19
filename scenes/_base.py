from manim import Scene

# Scenes read parameters through self.params with sensible defaults.
class ParamScene(Scene):
    def __init__(self, params=None, **kwargs):
        super().__init__(**kwargs)
        self.params = params or {}

    # Colorblind-safe palette (Okabe–Ito)
    def palette(self):
        return {
            "blue": "#0072B2",
            "orange": "#E69F00",
            "sky": "#56B4E9",
            "green": "#009E73",
            "yellow": "#F0E442",
            "red": "#D55E00",
            "purple": "#CC79A7",
            "black": "#000000",
            "gray": "#999999",
        }
