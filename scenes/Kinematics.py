from manim import *
# Works from CLI and from UI package
try:
    from ._base import ParamScene
except Exception:
    import os, sys, os.path as p
    sys.path.append(p.dirname(__file__))
    from _base import ParamScene

import numpy as np

# ---------- helpers ----------
def _nice_step(span, target=8):
    import math
    if span <= 0: return 1.0
    raw = span / max(1, target)
    exp = math.floor(math.log10(raw))
    frac = raw / (10 ** exp)
    nice = 1 if frac < 1.5 else (2 if frac < 3 else (5 if frac < 7 else 10))
    return nice * (10 ** exp)

def _add_number_labels_numberline(nline: NumberLine, start: float, end: float, step: float, color=GREY_B, font_scale=0.35):
    labels = VGroup()
    n = int(round((end - start) / step)) + 1
    for i in range(n):
        xv = start + i * step
        dot = Dot(nline.n2p(xv), radius=0.0001, color=color)
        lbl = Text(f"{xv:g}", color=color).scale(font_scale).next_to(dot, DOWN, buff=0.12)
        labels.add(lbl)
    return labels

def _add_axis_numbers_text(axes: Axes, x_vals=None, y_vals=None, color=GREY_B, fscale=0.35, buff=0.10):
    to_add = VGroup()
    x_axis = axes.get_x_axis()
    y_axis = axes.get_y_axis()
    if x_vals is not None:
        for xv in x_vals:
            p = x_axis.n2p(xv)
            to_add.add(Text(f"{xv:g}", color=color).scale(fscale).move_to(p + DOWN*0.25))
    if y_vals is not None:
        for yv in y_vals:
            p = y_axis.n2p(yv)
            to_add.add(Text(f"{yv:g}", color=color).scale(fscale).move_to(p + LEFT*0.35))
    return to_add

# ---------- UAM1D ----------
class UAM1D(ParamScene):
    """
    Params expected (all optional with sensible defaults):
      x0, v0, a, t_max, x_min, x_max, tick, show_xt, show_vt
    """
    def _load_params(self):
        if getattr(self, "params", None) and len(self.params):
            return dict(self.params)
        import json, os
        preset_key = os.getenv("UAM1D_PRESET", "UAM1D-CUSTOM")
        for path in (
            os.path.join(os.getcwd(), "presets", "uam1d_presets.json"),
            os.path.join(os.getcwd(), "uam1d_presets.json"),
        ):
            try:
                with open(path, "r") as f:
                    bank = json.load(f)
                if isinstance(bank, dict) and preset_key in bank:
                    return dict(bank[preset_key])
            except Exception:
                pass
        return {
            "x0": -5.0, "v0": 0.0, "a": 1.0, "t_max": 6.0,
            "x_min": -6.0, "x_max": 6.0, "tick": 1.0,
            "show_xt": False, "show_vt": False
        }

    @staticmethod
    def _first_positive_root(A, B, C):
        EPS = 1e-6
        if abs(A) < 1e-12:
            if abs(B) < 1e-12: return None
            t = -C / B
            return t if t > EPS else None
        D = B*B - 4*A*C
        if D < 0: return None
        sqrtD = D**0.5
        r1 = (-B - sqrtD) / (2*A)
        r2 = (-B + sqrtD) / (2*A)
        cands = [t for t in (r1, r2) if t > EPS]
        return min(cands) if cands else None

    def construct(self):
        P = self._load_params()
        p = self.palette()

        x0    = float(P.get("x0", -5.0))
        v0    = float(P.get("v0", 0.0))
        a     = float(P.get("a", 1.0))
        t_max = float(P.get("t_max", 6.0))
        x_min = float(P.get("x_min", -6.0))
        x_max = float(P.get("x_max",  6.0))
        tick  = float(P.get("tick", 1.0))
        show_xt = bool(P.get("show_xt", False))
        show_vt = bool(P.get("show_vt", False))

        title = Text("Uniformly Accelerated Motion (1D)", color=WHITE).scale(0.8).to_edge(UP)
        eq    = Text("x(t) = x0 + v0·t + 1/2·a·t²", color=WHITE).scale(0.6).next_to(title, DOWN)

        line  = NumberLine(
            x_range=[x_min, x_max, tick],
            length=10,
            include_numbers=False,
            color=GREY_B
        ).to_edge(DOWN)
        dot   = Dot(color=p["orange"]).scale(1.2).move_to(line.n2p(x0))
        labels = _add_number_labels_numberline(line, x_min, x_max, tick, color=GREY_B, font_scale=0.35)
        self.play(FadeIn(title), FadeIn(eq), Create(line), FadeIn(labels), FadeIn(dot))

        def x_of(t): return x0 + v0*t + 0.5*a*t*t
        def v_of(t): return v0 + a*t

        A = 0.5*a; B = v0
        t_hit_left  = self._first_positive_root(A, B, x0 - x_min)
        t_hit_right = self._first_positive_root(A, B, x0 - x_max)
        hits = [t for t in (t_hit_left, t_hit_right) if t is not None]
        t_stop = min(hits) if hits else t_max
        t_stop = min(t_stop, t_max)

        t = ValueTracker(0.0)
        dot.add_updater(lambda m: m.move_to(line.n2p(x_of(t.get_value()))))

        hud_z = 10
        time_text = always_redraw(lambda:
            Text(f"t = {t.get_value():.2f} s", color=WHITE)
                .scale(0.55).to_corner(UL).shift(DOWN*0.2 + RIGHT*0.2).set_z_index(hud_z)
        )
        vel_text = always_redraw(lambda:
            Text(f"v(t) = {v_of(t.get_value()):.2f} m/s", color=p["green"])
                .scale(0.55).next_to(time_text, DOWN, aligned_edge=LEFT, buff=0.15).set_z_index(hud_z)
        )
        acc_text = Text(f"a = {a:.2f} m/s²", color=p["red"]) \
            .scale(0.55).next_to(vel_text, DOWN, aligned_edge=LEFT, buff=0.15).set_z_index(hud_z)
        self.add(time_text, vel_text, acc_text)

        graphs = VGroup()
        if show_xt or show_vt:
            axis_style = dict(tips=False, axis_config={"include_numbers": False, "stroke_color": GREY_B})
            t_plot_max = max(1.0, float(t_stop))
            t_tick = max(1.0, round(t_plot_max/4, 1))

            if show_xt:
                ax_xt = Axes(
                    x_range=[0, t_plot_max, t_tick],
                    y_range=[x_min, x_max, tick],
                    x_length=5.2, y_length=3.2, **axis_style
                ).to_corner(UL, buff=0.6)
                xt_nums = _add_axis_numbers_text(ax_xt,
                    x_vals=np.arange(0, t_plot_max + 1e-9, t_tick),
                    y_vals=np.arange(x_min, x_max + 1e-9, tick),
                    color=GREY_B, fscale=0.30
                )
                xt_label = Text("x(t)", color=WHITE).scale(0.45).next_to(ax_xt, UP, buff=0.1)
                x_curve = always_redraw(lambda:
                    ax_xt.plot(lambda tt: x_of(tt), x_range=[0, t.get_value()], color=p["blue"])
                )
                x_marker = always_redraw(lambda:
                    Dot(color=p["orange"], radius=0.05).move_to(ax_xt.c2p(t.get_value(), x_of(t.get_value())))
                )
                graphs.add(ax_xt, xt_nums, xt_label, x_curve, x_marker)

            if show_vt:
                v_min = min(v0, v0 + a*t_plot_max) - 0.5
                v_max = max(v0, v0 + a*t_plot_max) + 0.5
                v_tick = max(1.0, round((v_max - v_min)/4, 1))
                ax_vt = Axes(
                    x_range=[0, t_plot_max, t_tick],
                    y_range=[v_min, v_max, v_tick],
                    x_length=5.2, y_length=3.2, **axis_style
                )
                if show_xt:
                    ax_vt.next_to(ax_xt, DOWN, buff=0.5).align_to(ax_xt, LEFT)
                else:
                    ax_vt.to_corner(UL, buff=0.6)
                vt_nums = _add_axis_numbers_text(ax_vt,
                    x_vals=np.arange(0, t_plot_max + 1e-9, t_tick),
                    y_vals=np.arange(v_min, v_max + 1e-9, v_tick),
                    color=GREY_B, fscale=0.30
                )
                vt_label = Text("v(t)", color=WHITE).scale(0.45).next_to(ax_vt, UP, buff=0.1)
                v_curve = always_redraw(lambda:
                    ax_vt.plot(lambda tt: v_of(tt), x_range=[0, t.get_value()], color=p["green"])
                )
                v_marker = always_redraw(lambda:
                    Dot(color=p["orange"], radius=0.05).move_to(ax_vt.c2p(t.get_value(), v_of(t.get_value())))
                )
                graphs.add(ax_vt, vt_nums, vt_label, v_curve, v_marker)

            self.play(FadeIn(graphs))

        self.wait(0.2)
        self.play(t.animate.set_value(t_stop), run_time=t_stop, rate_func=linear)
        caption = Text(f"x0={x0} m, v0={v0} m/s, a={a} m/s²", color=WHITE)\
            .scale(0.5).next_to(line, UP).set_z_index(10)
        self.play(Write(caption))
        self.wait(0.4)

class FreeFallSideBySide(ParamScene):
    """
    Params: y0=3, v_throw=6, g=9.8, t_max=2
    Dropped (v0=0) vs. Thrown Up (v0=+v_throw) from same height.
    """
    def construct(self):
        import os, json
        try:
            _env = os.getenv("PARAMS_JSON") or os.getenv("SCENE_PARAMS_JSON")
            if _env:
                payload = json.loads(_env)
                if not isinstance(getattr(self, "params", None), dict):
                    self.params = {}
                self.params.update(payload)
        except Exception:
            pass

        p = self.palette()
        y0   = float(self.params.get("y0", 3.0))
        v_th = float(self.params.get("v_throw", 6.0))
        g    = float(self.params.get("g", 9.8))
        tmax = float(self.params.get("t_max", 2.0))

        y_apex   = y0 + (v_th * v_th) / (2.0 * g) if v_th > 0 else y0
        y_top    = max(4.0, y_apex + 1.0)
        y_margin = max(1.5, 0.20 * y_top)
        y_step   = max(1.0, _nice_step(y_top))

        frame = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-y_margin, y_top, y_step],
            background_line_style={"stroke_opacity": 0.15},
        )

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-y_margin, y_top, y_step],
            tips=False,
            x_length=frame.x_length, y_length=frame.y_length,
            axis_config={"stroke_color": p["gray"], "include_ticks": True, "include_numbers": False, "tick_size": 0.05},
        )
        axes.move_to(frame.get_center())

        title  = Text("Free Fall: Dropped vs. Thrown Up").scale(0.8).to_edge(UP)
        x_axis = axes.get_x_axis().set_color(p["gray"]).set_z_index(3)
        y_axis = axes.get_y_axis().set_color(p["gray"]).set_z_index(3)

        y_vals = np.arange(-y_margin, y_top + 1e-9, y_step)
        y_nums = VGroup(*[
            Text(f"{yv:g}", color=p["gray"]).scale(0.35).move_to(y_axis.n2p(yv) + LEFT*0.35) for yv in y_vals
        ])
        self.play(FadeIn(frame), FadeIn(x_axis), FadeIn(y_axis), FadeIn(y_nums), FadeIn(title))

        def y_of_t(v0, t): return y0 + v0*t - 0.5*g*t*t
        def t_hit(v0):
            A = -0.5*g; B = v0; C = y0
            D = B*B - 4*A*C
            if D < 0: return None
            r1 = (-B - D**0.5)/(2*A); r2 = (-B + D**0.5)/(2*A)
            ts = [t for t in (r1, r2) if t > 1e-6]
            return min(ts) if ts else None

        t_thrown = t_hit(v_th)
        t_stop = min(t_thrown if t_thrown is not None else tmax, tmax)

        dropped = Dot(color=p["orange"]).move_to(frame.c2p(-1, y0))
        thrown  = Dot(color=p["sky"]).move_to(frame.c2p(+1, y0))
        ld = Text("Dropped").scale(0.5).next_to(dropped, UP, buff=0.2)
        lt = Text("Thrown Up").scale(0.5).next_to(thrown, UP, buff=0.2)
        self.play(FadeIn(dropped), FadeIn(thrown), FadeIn(ld), FadeIn(lt))

        t = ValueTracker(0.0)
        dropped.add_updater(lambda m: m.move_to(frame.c2p(-1, max(0, y_of_t(0,    t.get_value())))))
        thrown .add_updater(lambda m: m.move_to(frame.c2p(+1, max(0, y_of_t(v_th, t.get_value())))))

        hud = always_redraw(lambda:
            VGroup(
                Text(f"t = {t.get_value():.2f} s").scale(0.55),
                Text(f"v_dropped = {-g*t.get_value():.2f} m/s", color=p["green"]).scale(0.55),
                Text(f"v_thrown = {(v_th - g*t.get_value()):.2f} m/s", color=p["green"]).scale(0.55),
                Text(f"g = {g:.1f} m/s^2", color=p["red"]).scale(0.55),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UL)
        )
        self.add(hud)

        self.play(t.animate.set_value(t_stop), run_time=t_stop, rate_func=linear)
        self.wait(0.4)

class ProjectileMotion(ParamScene):
    """
    Params (optional): v0, theta_deg, y0, g, t_max
    Uses Axes as the single source of truth; NumberPlane is decorative only.
    """
    @staticmethod
    def _nice_step(max_val: float) -> float:
        import math
        if max_val <= 0:
            return 1.0
        target_ticks = 6
        rough = max_val / target_ticks
        exp = math.floor(math.log10(rough))
        base = 10 ** exp
        for k in (1, 2, 5, 10):
            if rough <= k * base:
                return k * base
        return 10 * base

    def construct(self):
        import os, json, math
        try:
            payload = os.getenv("PARAMS_JSON") or os.getenv("SCENE_PARAMS_JSON")
            if payload:
                payload = json.loads(payload)
                if not isinstance(getattr(self, "params", None), dict):
                    self.params = {}
                self.params.update(payload)
        except Exception:
            pass

        p = self.palette()
        v0     = float(self.params.get("v0", 10.0))
        th_deg = float(self.params.get("theta_deg", 45.0))
        y0     = float(self.params.get("y0", 0.0))
        g      = float(self.params.get("g", 9.8))
        tmax   = float(self.params.get("t_max", 5.0))

        th  = math.radians(th_deg)
        vx, vy0 = v0*math.cos(th), v0*math.sin(th)

        A = -0.5*g; B = vy0; C = y0
        disc = B*B - 4*A*C
        t_flight = tmax
        if disc >= 0:
            r1 = (-B - math.sqrt(disc)) / (2*A)
            r2 = (-B + math.sqrt(disc)) / (2*A)
            pos = [t for t in (r1, r2) if t > 1e-9]
            if pos:
                t_flight = min(max(pos), tmax)

        x_stop = max(0.0, vx * t_flight)
        y_apex = y0 + (vy0*vy0)/(2*g) if vy0 > 0 else y0
        y_top  = max(2.0, y_apex * 1.05 + 0.5)

        x_step = max(0.5, self._nice_step(x_stop if x_stop > 0 else 5.0))
        y_step = max(0.5, self._nice_step(y_top))

        def snap_up(val, step):
            import math
            return step * math.ceil(val / step + 1e-9)

        x_max = snap_up(max(4.0, x_stop * 1.02), x_step)
        y_max = snap_up(y_top, y_step)
        y_min = 0.0

        axes = Axes(
            x_range=[0, x_max, x_step],
            y_range=[y_min, y_max, y_step],
            x_length=10, y_length=6,
            tips=False,
            axis_config={
                "include_ticks": True,
                "include_numbers": False,   # <-- no LaTeX
                "tick_size": 0.06,
                "stroke_color": p["gray"],
            },
        )

        plane = NumberPlane(
            x_range=[0, x_max, x_step],
            y_range=[y_min, y_max, y_step],
            background_line_style={"stroke_opacity": 0.15},
            x_length=axes.x_length, y_length=axes.y_length,
        )
        plane.move_to(axes.get_center())

        x_vals = np.arange(0, x_max + 1e-9, x_step)
        y_vals = np.arange(y_min, y_max + 1e-9, y_step)
        axis_nums = _add_axis_numbers_text(axes, x_vals=x_vals, y_vals=y_vals, color=p["gray"], fscale=0.35)

        x_label = Text("x (m)").set_color(p["gray"]).scale(0.6).next_to(axes.get_x_axis(), DOWN, buff=0.35)
        y_label = Text("y (m)").set_color(p["gray"]).scale(0.6).next_to(axes.get_y_axis(), LEFT, buff=0.35)
        title_txt = "Projectile Motion — Level Launch" if abs(th_deg) < 1e-6 \
                    else f"Projectile Motion — Angled Launch (θ={th_deg:.1f}°)"
        title = Text(title_txt).scale(0.8).to_edge(UP)

        self.play(FadeIn(plane), Create(axes), FadeIn(axis_nums), FadeIn(x_label), FadeIn(y_label), FadeIn(title))

        def x_of(t): return vx*t
        def y_of(t): return y0 + vy0*t - 0.5*g*t*t

        t = ValueTracker(0.0)
        dot = Dot(color=p["orange"]).move_to(axes.c2p(0, y0))
        trail = TracedPath(dot.get_center, stroke_color=p["purple"], stroke_opacity=0.85, stroke_width=3)

        def clamp_ground(y):
            return max(y_min, y)

        dot.add_updater(lambda m: m.move_to(axes.c2p(x_of(t.get_value()), clamp_ground(y_of(t.get_value())))))
        self.add(trail, dot)

        hud = always_redraw(lambda:
            VGroup(
                Text(f"t = {t.get_value():.2f} s").scale(0.55),
                Text(f"v_x = {vx:.2f} m/s", color=p["green"]).scale(0.55),
                Text(f"v_y = {(vy0 - g*t.get_value()):.2f} m/s", color=p["green"]).scale(0.55),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UL)
        )
        self.add(hud)

        self.play(t.animate.set_value(t_flight), run_time=t_flight, rate_func=linear)
        self.wait(0.4)
