from manim import *
import math, os, json, numpy as np
from types import SimpleNamespace

# ------------------------------------------------------------------
# Minimal ParamScene fallback
# ------------------------------------------------------------------
class ParamScene(Scene):
    params: dict = {}
    def palette(self):
        # Colorblind-safe palette
        return SimpleNamespace(
            obj=GRAY_B, axis=GRAY_D, tick=GRAY_C, text=WHITE,
            force=BLUE_C, fric=ORANGE, normal=GREEN_C, weight=RED_C, tension=PURPLE_C,
            good=TEAL_C, warn=YELLOW, bad=RED_E, gray=GRAY_B, sky=BLUE_C, purple=PURPLE_C, green=GREEN_C, red=RED_C, orange=ORANGE
        )
    def _merge_env_params(self):
        try:
            _env = os.getenv("PARAMS_JSON") or os.getenv("SCENE_PARAMS_JSON")
            if _env:
                payload = json.loads(_env)
                if not isinstance(getattr(self, "params", None), dict):
                    self.params = {}
                self.params.update(payload)
        except Exception:
            pass

# ===================== 1) INCLINED PLANE (No LaTeX) =====================
class InclinedPlane(ParamScene):
    """
    Params:
      m=1.5, theta_deg=25, mu_s=0.50, mu_k=0.40, F_along=0.0, g=9.8, t_max=2.5,
      mode='student', show_time=False
    """
    def construct(self):
        import numpy as np

        self._merge_env_params()
        p = self.palette()

        # --- Params ---
        m       = float(self.params.get("m", 1.5))
        th_deg  = float(self.params.get("theta_deg", 25.0))
        th      = math.radians(th_deg)
        mu_s    = float(self.params.get("mu_s", 0.50))
        mu_k    = float(self.params.get("mu_k", 0.40))
        Fp      = float(self.params.get("F_along", 0.0))
        g       = float(self.params.get("g", 9.8))
        tmax    = float(self.params.get("t_max", 2.5))
        mode    = str(self.params.get("mode", "student"))
        show_time = bool(self.params.get("show_time", False))

        # --- Ramp geometry ---
        base_y = -2.0
        baseL = np.array([-4.0, base_y, 0.0])
        baseR = np.array([ 4.0, base_y, 0.0])
        height = (baseR[0] - baseL[0]) * math.tan(th)
        topR  = np.array([ baseR[0], base_y + height, 0.0 ])
        ramp  = Polygon(baseL, baseR, topR, color=p.axis)
        axis  = Line(baseL, topR, color=p.axis)

        # Unit vector along ramp (low→high)
        along_dir = (axis.get_end() - axis.get_start())
        along_dir = along_dir / np.linalg.norm(along_dir)

        # --- Block ---
        block = Square(1, color=p.obj, fill_color=p.obj, fill_opacity=0.2)
        block.move_to(axis.point_from_proportion(0.70))
        block.rotate(th)

        # --- Dynamics ---
        N       = m * g * math.cos(th)
        mg_sin  = m * g * math.sin(th)
        thr     = mu_s * N
        need    = abs(Fp - mg_sin)
        static  = (need <= thr)
        sgn = 1 if (Fp - mg_sin) > 0 else -1

        if static:
            a = 0.0
            f = need
        else:
            f = mu_k * N
            a = (Fp - mg_sin - sgn * f) / m

        # --- Force vectors ---
        g_vec = always_redraw(lambda:
            Arrow(block.get_center(), block.get_center() + DOWN * 1.6, color=p.weight)
        )
        N_vec = always_redraw(lambda:
            Arrow(block.get_center(), block.get_center() + UP * 1.2, color=p.normal
            ).rotate(th, about_point=block.get_center())
        )
        Fp_vec = always_redraw(lambda:
            Arrow(block.get_center() - along_dir * 0.6,
                  block.get_center() + along_dir * 0.6,
                  color=p.force).scale((abs(Fp) + 1) / 6)
        )
        f_vec = always_redraw(lambda:
            Arrow(block.get_center() + along_dir * 0.6 * sgn,
                  block.get_center() - along_dir * 0.6 * sgn,
                  color=p.fric).scale((abs(f) + 1) / 6)
        )

        # --- Header / Time (Text) ---
        if mode == "teacher":
            header = Text(f"θ={th_deg:.0f}°, N={N:.2f}, |Fp - mg·sinθ|={need:.2f}, μsN={thr:.2f} ⇒ a={a:.2f} m/s²",
                          font_size=30, color=p.text).to_edge(UP)
        else:
            header = Text("Given: m, θ, μs, μk, F∥, g", font_size=30, color=p.text).to_edge(UP)

        t_tracker = ValueTracker(0.0)
        if show_time:
            t_disp = always_redraw(lambda:
                Text(f"t = {t_tracker.get_value():.2f} s", font_size=28, color=p.text).to_corner(UL).shift(DOWN*0.1 + RIGHT*0.1)
            )

        badge = Text("STATIC" if static else "SLIDING", font_size=30, color=p.text).to_corner(UR).shift(DOWN*0.5)

        self.play(Create(ramp), Create(axis), FadeIn(block))
        self.play(GrowArrow(g_vec), GrowArrow(N_vec), GrowArrow(Fp_vec), GrowArrow(f_vec), FadeIn(header), FadeIn(badge))
        if show_time: self.play(FadeIn(t_disp))

        if not static and abs(a) > 1e-10:
            base = np.array(axis.get_start())
            end  = np.array(axis.get_end())
            L    = np.linalg.norm(end - base)

            u0 = float(np.dot(np.array(block.get_center()) - base, along_dir))

            margin = 0.6
            u_min, u_max = margin, L - margin

            def clamp_u(u):
                return max(u_min, min(u_max, u))

            def place_by_time(mobj):
                t = t_tracker.get_value()
                u = clamp_u(u0 + 0.5 * a * (t ** 2))
                mobj.move_to(base + along_dir * u)

            block.add_updater(place_by_time)

            du_cap = (u_max - u0) if a >= 0 else (u0 - u_min)
            t_stop_cap = math.sqrt(max(0.0, 2.0 * du_cap / abs(a)))
            run_t = min(tmax, t_stop_cap)

            self.play(t_tracker.animate.set_value(run_t), run_time=run_t, rate_func=linear)
            block.remove_updater(place_by_time)

        self.wait(0.4)

class AtwoodMachine(ParamScene):
    """
    Params:
      m1=1.0, m2=1.4, R=0.05, I=0.0, tau_drag=0.0, g=9.8, t_max=2.5,
      mode='student', show_time=True
    """
    def construct(self):
        import numpy as np

        self._merge_env_params()
        p = self.palette()

        m1   = float(self.params.get("m1", 1.0))
        m2   = float(self.params.get("m2", 1.4))
        Rm   = float(self.params.get("R", 0.05))
        I    = float(self.params.get("I", 0.0))
        tau  = float(self.params.get("tau_drag", 0.0))
        g    = float(self.params.get("g", 9.8))
        tmax = float(self.params.get("t_max", 2.5))
        mode = str(self.params.get("mode", "student"))
        show_time = bool(self.params.get("show_time", True))

        r = max(0.25, min(0.9, 8.0 * Rm))
        C = np.array([0.0, 1.6, 0.0])
        pulley = Circle(radius=r, color=p.axis).move_to(C)
        left_anchor  = C + LEFT  * r
        right_anchor = C + RIGHT * r

        ground_y = -3.2
        ground   = Line(LEFT*6 + UP*ground_y, RIGHT*6 + UP*ground_y, color=p.axis)

        vr_x   = C[0] + r + 0.9
        y_top  = C[1] - r
        y_bot  = ground_y
        tick_v_step = 0.5
        v_line  = Line([vr_x, y_bot, 0], [vr_x, y_top, 0], color=p.tick, stroke_width=2)
        v_ticks = VGroup(); v_labels = VGroup()
        yv = y_bot; val = 0.0
        while yv <= y_top + 1e-6:
            pnt = np.array([vr_x, yv, 0])
            v_ticks.add(Line(pnt+LEFT*0.08, pnt+RIGHT*0.08, color=p.tick, stroke_width=2))
            lab = Text(f"{val:.1f}", font_size=22, color=p.text)
            lab.move_to([vr_x - 0.25, yv, 0])
            v_labels.add(lab)
            yv += tick_v_step; val += tick_v_step

        mobj1 = Square(0.6, color=p.obj, fill_color=p.obj, fill_opacity=0.2).move_to(left_anchor  + DOWN*(r + 1.7))
        mobj2 = Square(0.7, color=p.obj, fill_color=p.obj, fill_opacity=0.2).move_to(right_anchor + DOWN*(r + 1.5))

        ropeL = always_redraw(lambda: Line(left_anchor,  mobj1.get_top(),  color=p.axis))
        ropeR = always_redraw(lambda: Line(right_anchor, mobj2.get_top(), color=p.axis))

        R_safe = max(Rm, 1e-6)
        denom  = (m1 + m2 + (I / max(R_safe**2, 1e-6)))
        a      = ( (m2 - m1) * g - (tau / R_safe) ) / denom
        if a >= 0:
            T2, T1 = m2*(g - a), m1*(g + a)
        else:
            T2, T1 = m2*(g + abs(a)), m1*(g - abs(a))

        if mode == "teacher":
            header = Text(f"a={a:.2f}, T1≈{T1:.1f}, T2≈{T2:.1f}", font_size=34, color=p.text).to_edge(UP)
        else:
            header = Text("Given: m1, m2, R, I, τ, g", font_size=34, color=p.text).to_edge(UP)

        t_tracker = ValueTracker(0.0)
        if show_time:
            t_disp = always_redraw(lambda:
                Text(f"t = {t_tracker.get_value():.2f} s", font_size=28, color=p.text).to_corner(UL).shift(DOWN*0.1+RIGHT*0.1)
            )

        self.play(Create(v_line), Create(v_ticks), FadeIn(v_labels))
        self.play(Create(pulley), Create(ground), FadeIn(mobj1), FadeIn(mobj2),
                  Create(ropeL), Create(ropeR), FadeIn(header))
        if show_time: self.play(FadeIn(t_disp))

        if abs(a) > 1e-8:
            margin   = 0.08
            top_cap  = C[1] - r - margin
            bot_cap  = ground_y + margin

            x1, y1_0 = mobj1.get_center()[0], mobj1.get_center()[1]
            x2, y2_0 = mobj2.get_center()[0], mobj2.get_center()[1]

            h1_top  = mobj1.get_top()[1]  - y1_0
            h1_bot  = y1_0 - mobj1.get_bottom()[1]
            h2_top  = mobj2.get_top()[1]  - y2_0
            h2_bot  = y2_0 - mobj2.get_bottom()[1]

            y1_max = top_cap - h1_top
            y1_min = bot_cap + h1_bot
            y2_max = top_cap - h2_top
            y2_min = bot_cap + h2_bot

            if a >= 0:
                d1 = max(0.0, y1_max - y1_0)
                d2 = max(0.0, y2_0 - y2_min)
            else:
                d1 = max(0.0, y1_0 - y1_min)
                d2 = max(0.0, y2_max - y2_0)

            d_cap  = min(d1, d2)
            t_stop = min(tmax, math.sqrt(2.0 * d_cap / abs(a)) if d_cap > 0 else 0.0)

            def clamp(v, lo, hi): return max(lo, min(hi, v))
            def place_m1(m):
                t = t_tracker.get_value()
                y = clamp(y1_0 + 0.5 * a * (t**2), y1_min, y1_max)
                m.move_to([x1, y, 0])
            def place_m2(m):
                t = t_tracker.get_value()
                y = clamp(y2_0 - 0.5 * a * (t**2), y2_min, y2_max)
                m.move_to([x2, y, 0])

            mobj1.add_updater(place_m1)
            mobj2.add_updater(place_m2)

            self.play(t_tracker.animate.set_value(t_stop), run_time=t_stop, rate_func=linear)

            mobj1.remove_updater(place_m1)
            mobj2.remove_updater(place_m2)

        self.wait(0.4)

class HalfAtwood(ParamScene):
    """
    Params:
      m_table=1.2, m_hanging=0.8, mu_s=0.30, mu_k=0.25, R=0.05, g=9.8, t_max=2.5,
      table_y=-0.8, mode='student', show_time=True
    """
    def construct(self):
        import numpy as np

        self._merge_env_params()
        p = self.palette()

        m1   = float(self.params.get("m_table", 1.2))
        m2   = float(self.params.get("m_hanging", 0.8))
        mu_s = float(self.params.get("mu_s", 0.30))
        mu_k = float(self.params.get("mu_k", 0.25))
        Rm   = float(self.params.get("R", 0.05))
        g    = float(self.params.get("g", 9.8))
        tmax = float(self.params.get("t_max", 2.5))
        table_y = float(self.params.get("table_y", -0.8))
        mode  = str(self.params.get("mode", "student"))
        show_time = bool(self.params.get("show_time", True))

        r = max(0.25, min(0.9, 8.0 * Rm))
        table_th = 0.22
        xL, xR = -5.2, 2.8

        table = Rectangle(width=(xR-xL), height=table_th, color=p.axis).move_to(
            np.array([(xL+xR)/2, table_y + table_th/2, 0])
        )
        ground_y = -3.2
        ground = Line(LEFT*6 + UP*ground_y, RIGHT*6 + UP*ground_y, color=p.axis)

        C = np.array([xR + r, table_y + table_th + r + 0.02, 0.0])
        pulley = Circle(radius=r, color=p.axis).move_to(C)
        left_tan   = C + LEFT  * r
        bottom_tan = C + DOWN  * r

        tick_h_step = 1.0
        ruler_y = table_y + table_th + 0.04
        h_line = Line([xL, ruler_y, 0], [xR, ruler_y, 0], color=p.tick, stroke_width=2)
        h_ticks = VGroup()
        for xv in range(int(math.ceil(xL)), int(math.floor(xR))+1, int(tick_h_step)):
            pnt = np.array([xv, ruler_y, 0])
            h_ticks.add(Line(pnt+UP*0.08, pnt+DOWN*0.08, color=p.tick, stroke_width=2))
        horiz_ruler = VGroup(h_line, h_ticks)

        tick_v_step = 0.5
        y_top = table_y + table_th
        y_bot = ground_y
        vr_x = C[0] + r + 0.35
        v_line = Line([vr_x, y_bot, 0], [vr_x, y_top, 0], color=p.tick, stroke_width=2)
        v_ticks = VGroup()
        yv = y_bot
        while yv <= y_top + 1e-6:
            pnt = np.array([vr_x, yv, 0])
            v_ticks.add(Line(pnt+LEFT*0.08, pnt+RIGHT*0.08, color=p.tick, stroke_width=2))
            yv += tick_v_step
        vert_ruler = VGroup(v_line, v_ticks)

        cart = Rectangle(width=1.6, height=0.9, color=p.obj, fill_color=p.obj, fill_opacity=0.2)
        cart.move_to(np.array([ (xL+xR)/2 - 1.2, table_y + table_th + cart.height/2, 0 ]))
        hanger = Square(0.9, color=p.obj, fill_color=p.obj, fill_opacity=0.2)
        hanger.move_to(bottom_tan + DOWN*(r + 1.2))

        def cart_anchor_x(): return cart.get_right()[0]
        rope_left  = always_redraw(lambda: Line(np.array([cart_anchor_x(), left_tan[1], 0.0]),
                                                left_tan, color=p.axis))
        rope_right = always_redraw(lambda: Line(bottom_tan, hanger.get_top(), color=p.axis))

        static = (m2 * g) <= (mu_s * m1 * g)
        a = 0.0 if static else (m2 * g - mu_k * m1 * g) / (m1 + m2)

        if mode == "teacher":
            header = Text(f"a = {a:.2f} m/s²", font_size=34, color=p.text).to_edge(UP)
        else:
            header = Text("Given: m_table, m_hang, μs, μk, R, g", font_size=34, color=p.text).to_edge(UP)

        t_tracker = ValueTracker(0.0)
        if show_time:
            t_disp = always_redraw(lambda:
                Text(f"t = {t_tracker.get_value():.2f} s", font_size=28, color=p.text).to_corner(UL).shift(DOWN*0.1+RIGHT*0.1)
            )
        cart_x0 = cart.get_center()[0]
        hang_y0 = hanger.get_center()[1]

        dx_disp = always_redraw(lambda:
            Text(f"|Δx| = {abs(cart.get_center()[0] - cart_x0):.2f} m", font_size=28, color=p.text).to_corner(UL).shift(DOWN*0.9+RIGHT*0.1)
        )
        dy_disp = always_redraw(lambda:
            Text(f"|Δy| = {abs(hang_y0 - hanger.get_center()[1]):.2f} m", font_size=28, color=p.text).to_corner(UL).shift(DOWN*1.5+RIGHT*0.1)
        )

        self.play(Create(horiz_ruler), Create(vert_ruler))
        self.play(Create(table), Create(ground), Create(pulley),
                  FadeIn(cart), FadeIn(hanger), Create(rope_left), Create(rope_right),
                  FadeIn(header))
        if show_time: self.play(FadeIn(t_disp))
        self.play(FadeIn(dx_disp), FadeIn(dy_disp))

        if abs(a) > 1e-8:
            margin = 0.08
            right_edge_limit = (left_tan[0] - margin)
            left_edge_limit  = (xL + margin)
            cx_min = left_edge_limit  + cart.width/2
            cx_max = right_edge_limit - cart.width/2

            up_cap_y    = bottom_tan[1] - margin
            down_cap_y  = ground_y + margin
            hy_min = down_cap_y + (hanger.height/2)
            hy_max = up_cap_y   - (hanger.height/2)

            x0 = cart.get_center()[0]
            y0 = hanger.get_center()[1]

            if a >= 0:
                d_cart = max(0.0, cx_max - x0)
                d_hang = max(0.0, y0 - hy_min)
            else:
                d_cart = max(0.0, x0 - cx_min)
                d_hang = max(0.0, hy_max - y0)

            d_cap  = min(d_cart, d_hang)
            t_stop = min(tmax, math.sqrt(2.0 * d_cap / abs(a)) if d_cap > 0 else 0.0)

            def clamp(v, lo, hi): return max(lo, min(hi, v))
            y_cart_fixed = cart.get_center()[1]
            x_hang_fixed = hanger.get_center()[0]

            def place_cart(m):
                t = t_tracker.get_value()
                x = clamp(x0 + 0.5 * a * (t**2), cx_min, cx_max)
                m.move_to([x, y_cart_fixed, 0])
            def place_hanger(m):
                t = t_tracker.get_value()
                y = clamp(y0 - 0.5 * a * (t**2), hy_min, hy_max)
                m.move_to([x_hang_fixed, y, 0])

            cart.add_updater(place_cart)
            hanger.add_updater(place_hanger)

            self.play(t_tracker.animate.set_value(t_stop), run_time=t_stop, rate_func=linear)

            cart.remove_updater(place_cart)
            hanger.remove_updater(place_hanger)

        self.wait(0.4)
