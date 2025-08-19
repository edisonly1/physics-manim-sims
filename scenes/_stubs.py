from manim import *
from ._base import ParamScene

def make_stub(class_name: str, title: str, defaults: dict | None = None):
    """Return a minimal, safe-to-render placeholder scene."""
    defaults = defaults or {}
    class _Stub(ParamScene):
        __doc__ = f"Params (suggested defaults): {defaults}"
        def construct(self):
            p = self.palette()
            banner = Text(title).scale(0.7).to_edge(UP)
            note = Text("Placeholder scene — implement visuals next.")
            note.scale(0.5).next_to(banner, DOWN)
            box = Rectangle(width=10, height=5, color=p["gray"]).set_stroke(width=2)
            self.play(FadeIn(banner), FadeIn(note), Create(box))
            self.wait(0.5)
    _Stub.__name__ = class_name
    return _Stub
