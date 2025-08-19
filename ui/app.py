# app.py — multipage UI for Kinematics + Forces (auto-preset banks)
import os, json, shutil, subprocess
from pathlib import Path
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, field_validator

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
RENDERS_DIR = Path("renders"); RENDERS_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_DIR = Path("presets"); PRESETS_DIR.mkdir(parents=True, exist_ok=True)

APP_DIR = Path(__file__).resolve().parent

def resolve_scene_file(fname: str) -> str:
    """Search common locations for the scene file."""
    env_scene_dir = os.getenv("SCENE_DIR")
    extra = []
    if env_scene_dir:
        extra.append(Path(env_scene_dir)/Path(fname).name)
    cands = [
        Path(fname),
        APP_DIR/Path(fname),
        APP_DIR/Path("..")/Path(fname),
        Path("scenes")/Path(fname).name,
        APP_DIR/Path("scenes")/Path(fname).name,
        APP_DIR/Path("..")/Path("scenes")/Path(fname).name,
        Path(Path(fname).name),
    ] + extra
    for c in cands:
        if c.exists():
            return str(c.resolve())
    # Fall back to provided name; manim will error clearly if missing
    return fname

def load_preset_bank(path: Path, default_key: str, default_params: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            st.warning(f"Preset file {path} unreadable—starting fresh.")
    # initialize with a default key when file doesn't exist
    bank = {default_key: default_params}
    try:
        path.write_text(json.dumps(bank, indent=2))  # auto-create file like kinematics
    except Exception:
        pass
    return bank

def save_preset_bank(path: Path, bank: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, indent=2))

def manim_render(scene_file: str, scene_class: str, out_stub: str, quality: str, env: dict) -> Path:
    sf = resolve_scene_file(scene_file)
    sf_abs = str(Path(sf).resolve())
    cmd = ["manim", f"-q{quality}", sf_abs, scene_class, "-o", f"{out_stub}.mp4"]
    with st.status("Rendering with Manim…", expanded=True) as status:
        st.write("• Running:", " ".join(cmd))
        if "UAM1D_PRESET" in env:
            st.write("• Using preset key:", env["UAM1D_PRESET"])
        if "PARAMS_JSON" in env:
            st.code(env["PARAMS_JSON"], language="json")

        run_cwd = str(Path(sf).resolve().parent)
        proc = subprocess.run(cmd, env=env, cwd=run_cwd, capture_output=True, text=True, timeout=900)

        # Show recent stdout and full stderr on error
        st.write(proc.stdout[-1500:] or "(no stdout)")
        if proc.returncode != 0:
            st.error("⚠️ Manim failed. See stderr below.")
            st.code(proc.stderr or "(empty stderr)")
            status.update(state="error", label="Render failed")
            return None
        status.update(label="Packaging output…")

    # Try to locate the produced video near the scene file first
    produced = None
    out_name = f"{out_stub}.mp4"
    run_root = Path(sf).resolve().parent
    for p in run_root.rglob(out_name):
        if "media/videos" in str(p):
            produced = p; break
    if not produced:
        cands = sorted(run_root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands: produced = cands[0]
    if not produced:
        # Fallback: search from current working directory
        cands = sorted(Path(".").rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands: produced = cands[0]
    if not produced:
        raise FileNotFoundError("Could not locate Manim output MP4.")

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    target = RENDERS_DIR / produced.name
    shutil.move(str(produced), target)
    return target

def render_paramscene_generic(scene_file: str, scene_class: str, out_name: str, params_dict: dict, quality: str) -> Path:
    env = os.environ.copy()
    payload = json.dumps(params_dict)
    env["PARAMS_JSON"] = payload
    env["SCENE_PARAMS_JSON"] = payload
    return manim_render(scene_file, scene_class, out_name.replace(" ", "_"), quality, env)

def render_uam1d(scene_file: str, out_name: str, params: "UAMParams", quality: str, preset_path: Path) -> Path:
    # keep "bank-based" render for UAM1D so Manim can pick from env key
    try:
        bank = json.loads(preset_path.read_text()) if preset_path.exists() else {}
    except Exception:
        bank = {}
    preset_key = "UAM1D-CUSTOM"
    bank[preset_key] = params.model_dump()
    save_preset_bank(preset_path, bank)
    env = os.environ.copy()
    env["UAM1D_PRESET"] = preset_key
    return manim_render(scene_file, "UAM1D", out_name.replace(" ", "_"), quality, env)

# -----------------------------------------------------------------------------
# Pydantic Param Models (Kinematics)
# -----------------------------------------------------------------------------
class Units(BaseModel):
    x: str = "m"
    t: str = "s"
    v: str = "m/s"
    a: str = "m/s²"

class UAMParams(BaseModel):
    a: float = Field(1.5, description="m/s^2")
    v0: float = Field(0.0, description="m/s")
    x0: float = Field(-4.0, description="m")
    x_min: float = -5.0
    x_max: float = 5.0
    tick: float = 1.0
    t_max: float = 6.0
    stop_at_edge: bool = True
    show_xt: bool = False
    show_vt: bool = False
    trail: bool = True
    fps: int = 30
    width: int = 1920
    height: int = 1080
    caption: str = "UAM: custom"
    units: Units = Units()
    @field_validator("x_max")
    @classmethod
    def check_bounds(cls, v, info):
        x_min = info.data.get("x_min", -5.0)
        if v <= x_min:
            raise ValueError("x_max must be > x_min")
        return v

class FreeFallParams(BaseModel):
    y0: float = 3.0
    v_throw: float = 6.0
    g: float = 9.8
    t_max: float = 2.0
    caption: str = "Free Fall: Dropped vs Thrown"

class ProjParams(BaseModel):
    v0: float = 8.0
    theta_deg: float = 35.0  # set 0 for level launch
    y0: float = 0.5
    g: float = 9.8
    t_max: float = 3.5
    caption: str = "Projectile Motion"

# -----------------------------------------------------------------------------
# Pydantic Param Models (Forces)
# -----------------------------------------------------------------------------
class ForcesFrictionParams(BaseModel):
    m: float = 2.0
    mu_s: float = 0.40
    mu_k: float = 0.30
    F: float = 7.0
    g: float = 9.8
    t_max: float = 2.5
    x_min: float = -1.0
    x_max: float = 7.0
    caption: str = "Forces & Friction (horizontal)"

class InclinedPlaneParams(BaseModel):
    m: float = 1.5
    theta_deg: float = 25.0
    mu_s: float = 0.50
    mu_k: float = 0.40
    F_along: float = 0.0
    g: float = 9.8
    t_max: float = 2.5
    # NEW:
    mode: str = "student"     # 'student' | 'teacher'
    show_time: bool = False
    caption: str = "Inclined Plane"

class AtwoodParams(BaseModel):
    m1: float = 1.0
    m2: float = 1.4
    R: float = 0.05
    I: float = 0.0
    tau_drag: float = 0.0
    g: float = 9.8
    t_max: float = 2.5
    # NEW:
    mode: str = "student"     # 'student' | 'teacher'
    show_time: bool = True
    caption: str = "Atwood Machine"

class HalfAtwoodParams(BaseModel):
    m_table: float = 1.2
    m_hanging: float = 0.8
    mu_s: float = 0.30
    mu_k: float = 0.25
    R: float = 0.05
    g: float = 9.8
    t_max: float = 2.5
    table_y: float = -0.8   # controls table height
    # NEW:
    mode: str = "student"     # 'student' | 'teacher'
    show_time: bool = True
    caption: str = "Half-Atwood Machine"

# -----------------------------------------------------------------------------
# Scene Registry
# -----------------------------------------------------------------------------
SCENES = {
    # --- Kinematics (Kinematics.py) ---
    "UAM1D (Uniform Accel. 1D)": dict(
        scene_file="Kinematics.py",
        scene_class="UAM1D",
        preset_path=PRESETS_DIR / "uam1d_presets.json",
        Model=UAMParams,
        uses_preset_bank=True,
        title="Physics Animator — UAM1D",
    ),
    "Free Fall — Dropped vs Thrown": dict(
        scene_file="Kinematics.py",
        scene_class="FreeFallSideBySide",
        preset_path=PRESETS_DIR / "freefall_presets.json",
        Model=FreeFallParams,
        uses_preset_bank=False,
        title="Physics Animator — Free Fall (SxS)",
    ),
    "Projectile Motion (level or angle)": dict(
        scene_file="Kinematics.py",
        scene_class="ProjectileMotion",
        preset_path=PRESETS_DIR / "proj_motion_presets.json",
        Model=ProjParams,
        uses_preset_bank=False,
        title="Physics Animator — Projectile Motion",
    ),

    # --- Forces (Forces.py) ---
    "Forces: Inclined Plane": dict(
        scene_file="Forces.py",
        scene_class="InclinedPlane",
        preset_path=PRESETS_DIR / "inclined_plane_presets.json",
        Model=InclinedPlaneParams,
        uses_preset_bank=False,
        title="Physics Animator — Inclined Plane",
    ),
    "Forces: Atwood Machine": dict(
        scene_file="Forces.py",
        scene_class="AtwoodMachine",
        preset_path=PRESETS_DIR / "atwood_presets.json",
        Model=AtwoodParams,
        uses_preset_bank=False,
        title="Physics Animator — Atwood Machine",
    ),
    "Forces: Half-Atwood Machine": dict(
        scene_file="Forces.py",
        scene_class="HalfAtwood",
        preset_path=PRESETS_DIR / "half_atwood_presets.json",
        Model=HalfAtwoodParams,
        uses_preset_bank=False,
        title="Physics Animator — Half-Atwood",
    ),
}

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config("Physics Animator", page_icon="📽️", layout="wide")

st.sidebar.header("Animations")
page_name = st.sidebar.radio("Choose an animation", list(SCENES.keys()))

scene_meta = SCENES[page_name]
st.title(scene_meta["title"])

with st.expander("🔎 Preflight checks (click if renders fail)", expanded=False):
    # Show resolve result and manim path
    sf = resolve_scene_file(scene_meta["scene_file"])
    st.write("Scene file:", sf)
    try:
        import shutil as _sh
        manim_path = _sh.which("manim")
        st.write("manim executable:", manim_path or "(not found)")
    except Exception as _e:
        st.write("manim lookup error:", _e)

col_top1, col_top2 = st.columns([2,1])
with col_top1:
    out_name = st.text_input("Output name (no extension)", value=scene_meta["scene_class"])
with col_top2:
    quality = st.selectbox(
        "Quality",
        options=[("Preview (q=l)", "l"), ("Medium (q=m)", "m"), ("1080p (q=k)", "k")],
        index=0, format_func=lambda x: x[0]
    )[1]

Model = scene_meta["Model"]
preset_path: Path = scene_meta["preset_path"]
uses_bank = scene_meta["uses_preset_bank"]

default_key = f"{scene_meta['scene_class']}-Default"
default_params = Model().model_dump()
bank = load_preset_bank(preset_path, default_key, default_params)

st.sidebar.subheader("Presets")
preset_keys = list(bank.keys())
selected = st.sidebar.selectbox("Select preset", preset_keys, index=0)

with st.sidebar.expander("Save / Delete", expanded=False):
    new_name = st.text_input("New preset name", value=f"{scene_meta['scene_class']}-Custom-1")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save as new"):
            bank[new_name] = bank[selected]
            save_preset_bank(preset_path, bank)
            st.success(f"Saved {new_name}")
    with c2:
        if st.button("🗑️ Delete selected"):
            if selected in bank and len(bank) > 1:
                del bank[selected]
                save_preset_bank(preset_path, bank)
                st.success(f"Deleted {selected}")
            else:
                st.warning("Cannot delete last preset.")

st.subheader("Parameters")
session_key = f"current_params__{scene_meta['scene_class']}"
if session_key not in st.session_state:
    st.session_state[session_key] = Model(**bank[selected])

with st.form(f"param_form__{scene_meta['scene_class']}", clear_on_submit=False):
    p = Model(**bank[selected])

    # ---------------- Kinematics forms ----------------
    if scene_meta["scene_class"] == "UAM1D":
        c1, c2, c3 = st.columns(3)
        with c1:
            a = st.number_input("Acceleration a (m/s²)", value=float(p.a), step=0.1, format="%.3f")
            v0 = st.number_input("Initial velocity v₀ (m/s)", value=float(p.v0), step=0.1, format="%.3f")
            x0 = st.number_input("Initial position x₀ (m)", value=float(p.x0), step=0.1, format="%.3f")
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.5, min_value=0.5, format="%.2f")
            stop_at_edge = st.checkbox("Stop at number line edges", value=bool(p.stop_at_edge))
            trail = st.checkbox("Show trail", value=bool(p.trail))
        with c2:
            x_min = st.number_input("x_min (m)", value=float(p.x_min), step=0.5, format="%.2f")
            x_max = st.number_input("x_max (m)", value=float(p.x_max), step=0.5, format="%.2f")
            tick  = st.number_input("tick spacing (m)", value=float(p.tick), step=0.5, min_value=0.1, format="%.2f")
            show_xt = st.checkbox("Show x–t graph", value=bool(p.show_xt))
            show_vt = st.checkbox("Show v–t graph", value=bool(p.show_vt))
        with c3:
            fps = st.number_input("FPS", value=int(p.fps), min_value=12, max_value=60, step=6)
            width = st.number_input("Width (px)", value=int(p.width), min_value=640, step=160)
            height = st.number_input("Height (px)", value=int(p.height), min_value=360, step=90)
            caption = st.text_input("Caption", value=p.caption)
            ux = st.text_input("Units — x", value=p.units.x)
            ut = st.text_input("Units — t", value=p.units.t)
            uv = st.text_input("Units — v", value=p.units.v)
            ua = st.text_input("Units — a", value=p.units.a)

    elif scene_meta["scene_class"] == "FreeFallSideBySide":
        c1, c2 = st.columns(2)
        with c1:
            y0 = st.number_input("Initial height y₀ (m)", value=float(p.y0), step=0.1, format="%.2f")
            v_throw = st.number_input("Throw up speed (m/s)", value=float(p.v_throw), step=0.1, format="%.2f")
        with c2:
            g = st.number_input("g (m/s²)", value=float(p.g), step=0.1, format="%.2f")
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.1, format="%.2f")
        caption = st.text_input("Caption", value=p.caption)

    elif scene_meta["scene_class"] == "ProjectileMotion":
        c1, c2 = st.columns(2)
        with c1:
            v0 = st.number_input("v₀ (m/s)", value=float(p.v0), step=0.1, format="%.2f")
            theta_deg = st.number_input("θ (degrees) — set 0 for level", value=float(p.theta_deg), step=1.0, format="%.1f")
        with c2:
            y0 = st.number_input("Initial height y₀ (m)", value=float(p.y0), step=0.1, format="%.2f")
            g = st.number_input("g (m/s²)", value=float(p.g), step=0.1, format="%.2f")
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.1, format="%.2f")
        caption = st.text_input("Caption", value=p.caption)

    # ---------------- Forces forms ----------------
    elif scene_meta["scene_class"] == "ForcesFriction":
        c1, c2 = st.columns(2)
        with c1:
            m = st.number_input("Mass m (kg)", value=float(p.m), step=0.1)
            F = st.number_input("Applied force F (N)", value=float(p.F), step=0.5)
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.1)
        with c2:
            mu_s = st.number_input("μs", value=float(p.mu_s), step=0.01, format="%.2f")
            mu_k = st.number_input("μk", value=float(p.mu_k), step=0.01, format="%.2f")
            g = st.number_input("g (m/s²)", value=float(p.g), step=0.1)
            x_min = st.number_input("x_min (m)", value=float(p.x_min), step=0.5)
            x_max = st.number_input("x_max (m)", value=float(p.x_max), step=0.5)
        caption = st.text_input("Caption", value=p.caption)

    elif scene_meta["scene_class"] == "InclinedPlane":
        c1, c2 = st.columns(2)
        with c1:
            m = st.number_input("Mass m (kg)", value=float(p.m), step=0.1)
            theta_deg = st.number_input("θ (deg)", value=float(p.theta_deg), step=1.0)
            F_along = st.number_input("F_along (N)", value=float(p.F_along), step=0.5)
        with c2:
            mu_s = st.number_input("μs", value=float(p.mu_s), step=0.01)
            mu_k = st.number_input("μk", value=float(p.mu_k), step=0.01)
            g = st.number_input("g (m/s²)", value=float(p.g), step=0.1)
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.1)
        # NEW:
        mode_opt = st.radio("Mode", ["Student (hide result)", "Teacher (show result)"],
                            index=0 if p.mode=="student" else 1, horizontal=True)
        mode = "student" if "Student" in mode_opt else "teacher"
        show_time = st.checkbox("Show time counter", value=bool(p.show_time))
        caption = st.text_input("Caption", value=p.caption)

    elif scene_meta["scene_class"] == "AtwoodMachine":
        c1, c2, c3 = st.columns(3)
        with c1:
            m1 = st.number_input("m1 (kg)", value=float(p.m1), step=0.1)
            m2 = st.number_input("m2 (kg)", value=float(p.m2), step=0.1)
            g = st.number_input("g (m/s²)", value=float(p.g), step=0.1)
        with c2:
            R = st.number_input("Pulley radius R (m)", value=float(p.R), step=0.01, format="%.3f")
            I = st.number_input("Axle/pulley inertia I (kg·m²)", value=float(p.I), step=0.01)
            tau_drag = st.number_input("Axle drag τ (N·m)", value=float(p.tau_drag), step=0.01)
        with c3:
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.1)
        # NEW:
        mode_opt = st.radio("Mode", ["Student (hide a,T₁,T₂)", "Teacher (show a,T₁,T₂)"],
                            index=0 if p.mode=="student" else 1, horizontal=True)
        mode = "student" if "Student" in mode_opt else "teacher"
        show_time = st.checkbox("Show time counter", value=bool(p.show_time))
        caption = st.text_input("Caption", value=p.caption)

    elif scene_meta["scene_class"] == "HalfAtwood":
        c1, c2 = st.columns(2)
        with c1:
            m_table = st.number_input("m_table (kg)", value=float(p.m_table), step=0.1)
            m_hanging = st.number_input("m_hanging (kg)", value=float(p.m_hanging), step=0.1)
            R = st.number_input("Pulley radius R (m)", value=float(p.R), step=0.01, format="%.3f")
            table_y = st.number_input("Table height (y, up is +)", value=float(p.table_y), step=0.1)
        with c2:
            mu_s = st.number_input("μs", value=float(p.mu_s), step=0.01)
            mu_k = st.number_input("μk", value=float(p.mu_k), step=0.01)
            g = st.number_input("g (m/s²)", value=float(p.g), step=0.1)
            t_max = st.number_input("t_max (s)", value=float(p.t_max), step=0.1)
        # NEW:
        mode_opt = st.radio("Mode", ["Student (hide a)", "Teacher (show a)"],
                            index=0 if p.mode=="student" else 1, horizontal=True)
        mode = "student" if "Student" in mode_opt else "teacher"
        show_time = st.checkbox("Show time counter", value=bool(p.show_time))
        caption = st.text_input("Caption", value=p.caption)

    col_apply, col_save, col_render = st.columns([1,1,2])
    with col_apply:
        apply_btn = st.form_submit_button("Save to current")
    with col_save:
        save_btn = st.form_submit_button("Save overwrite")
    with col_render:
        render_now = st.form_submit_button("🎬 Apply & Render")

# Persist (and possibly render) after submit
if apply_btn or save_btn or render_now:
    try:
        # Build model instance per scene
        sc = scene_meta["scene_class"]
        if sc == "UAM1D":
            new_params = UAMParams(
                a=a, v0=v0, x0=x0, x_min=x_min, x_max=x_max, tick=tick, t_max=t_max,
                stop_at_edge=stop_at_edge, show_xt=show_xt, show_vt=show_vt, trail=trail,
                fps=fps, width=width, height=height, caption=caption,
                units=Units(x=ux, t=ut, v=uv, a=ua),
            )
        elif sc == "FreeFallSideBySide":
            new_params = FreeFallParams(y0=y0, v_throw=v_throw, g=g, t_max=t_max, caption=caption)
        elif sc == "ProjectileMotion":
            new_params = ProjParams(v0=v0, theta_deg=theta_deg, y0=y0, g=g, t_max=t_max, caption=caption)

        elif sc == "ForcesFriction":
            new_params = ForcesFrictionParams(m=m, mu_s=mu_s, mu_k=mu_k, F=F, g=g, t_max=t_max, x_min=x_min, x_max=x_max, caption=caption)
        elif sc == "InclinedPlane":
            new_params = InclinedPlaneParams(m=m, theta_deg=theta_deg, mu_s=mu_s, mu_k=mu_k,
                                             F_along=F_along, g=g, t_max=t_max,
                                             mode=mode, show_time=show_time, caption=caption)
        elif sc == "AtwoodMachine":
            new_params = AtwoodParams(m1=m1, m2=m2, R=R, I=I, tau_drag=tau_drag, g=g, t_max=t_max,
                                      mode=mode, show_time=show_time, caption=caption)
        elif sc == "HalfAtwood":
            new_params = HalfAtwoodParams(
                m_table=m_table, m_hanging=m_hanging, mu_s=mu_s, mu_k=mu_k,
                R=R, g=g, t_max=t_max, table_y=table_y,
                mode=mode, show_time=show_time, caption=caption
            )
        else:
            raise RuntimeError("Unknown scene.")

        st.session_state[session_key] = new_params
        if save_btn:
            bank[selected] = new_params.model_dump()
            save_preset_bank(preset_path, bank)
            st.success(f"Preset '{selected}' updated.")

        if render_now:
            if uses_bank and sc == "UAM1D":
                target = render_uam1d(scene_meta["scene_file"], out_name, new_params, quality, preset_path)
            else:
                target = render_paramscene_generic(scene_meta["scene_file"], sc, out_name, new_params.model_dump(), quality)
            if target is None:
                st.stop()
            st.success(f"Done: {target.name}")
            st.video(str(target))
            st.download_button("Download MP4", data=target.read_bytes(), file_name=target.name, mime="video/mp4")

    except ValidationError as e:
        st.error(str(e))

st.subheader("Render")
cR1, cR2 = st.columns([1,3])
with cR1:
    render_btn = st.button("🎬 Render clip", type="primary")
with cR2:
    st.markdown("Rendered files appear below and in `/renders`.")

if render_btn:
    try:
        current = st.session_state[session_key]
        sc = scene_meta["scene_class"]
        if uses_bank and sc == "UAM1D":
            target = render_uam1d(scene_meta["scene_file"], out_name, current, quality, preset_path)
        else:
            target = render_paramscene_generic(scene_meta["scene_file"], sc, out_name, current.model_dump(), quality)

        if target is None:
            st.stop()
        st.success(f"Done: {target.name}")
        st.video(str(target))
        st.download_button("Download MP4", data=target.read_bytes(), file_name=target.name, mime="video/mp4")
    except Exception as e:
        st.error(f"Render failed: {e}")

st.divider()
st.subheader("Previous renders")
files = sorted(RENDERS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
if files:
    for f in files:
        with st.expander(f.name):
            st.video(str(f))
            st.download_button("Download", data=f.read_bytes(), file_name=f.name, mime="video/mp4")
else:
    st.info("No renders yet.")
