# =============================================================================
# Quad-tank Open-loop Optimizer: a Streamlit tutorial app.
#
# This file builds an interactive web app for the quadruple-tank process: a
# classic chemical-engineering benchmark with two pumps feeding four tanks
# arranged in a 2x2 layout. Each pump's flow is split between a directly-fed
# lower tank and a diagonally-paired upper tank that drains into the *other*
# lower tank, producing the well-known cross-coupled dynamics.
#
# The solver runs an open-loop optimal control problem: starting from
# user-specified tank levels, find pump trajectories that drive all four
# tanks back to their steady states. This is a non-linear program (NLP)
# discretized with orthogonal collocation on finite elements. The model
# follows Raff et al. (2006) and the discretization follows Biegler (2010);
# both are cited from the in-app `📐 Formulation` tab.
#
# Library roadmap:
#   - streamlit : UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo     : algebraic modeling: sets, params, vars, constraints,
#                  objective. Continuous variables only (no integers).
#   - pounce    : the NLP solver, a Rust reimplementation of IPOPT
#                  (primal-dual interior-point). Called as a subprocess
#                  via Pyomo. Binary ships in the `pyomo-pounce` wheel.
#   - plotly    : both the animated schematic (Plotly frames + Play/Pause)
#                  and the time-series subplots.
#
# File roadmap:
#   1. Page config.
#   2. CSS / sidebar layout tweaks.
#   3. Sidebar widgets: initial tank heights, controller params, Solve button.
#   4. solve_model           : builds and solves the Pyomo NLP.
#   5. build_tank_figure     : assembles the animated process schematic.
#   6. build_timeseries      : small subplot grid of the optimized trajectories.
#   7. render_formulation_tab: static markdown for the Formulation tab.
#   8. Main layout           : manual-solve flow + four tabs (no auto-solve).
# =============================================================================

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pyomo.environ as pyo
# pounce: a Rust reimplementation of IPOPT, packaged as `pyomo-pounce`.
# Importing the package registers it with Pyomo's SolverFactory, so the
# `pyo.SolverFactory('pounce')` call later resolves to the bundled binary.
import pyomo_pounce  # noqa: F401
import io, contextlib, threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# `set_page_config` must be the first Streamlit call. Wide layout + open
# sidebar gives the schematic enough horizontal room.
st.set_page_config(page_title="Quad Tank", page_icon="favicon.png",
                   layout="wide", initial_sidebar_state="expanded")

# Solver: pounce (Rust reimplementation of IPOPT), shipped via the
# `pyomo-pounce` wheel, which bundles the solver binary: no system install
# required. Pyomo finds it through SolverFactory("pounce") below.

# Steady-state tank heights (cm): the reference point the controller drives
# back to. The optimization works in deviation variables (z = x - x_ss) but
# the UI displays absolute heights, so XSS is used to convert between them.
XSS = {1: 14.0, 2: 14.0, 3: 14.2, 4: 21.3}
MAX_H = 30.0  # display ceiling for all tanks (cm)

# ── Sidebar ──────────────────────────────────────────────────────────────────
# CSS overrides. The sidebar text gets `user-select: none` so dragging
# sliders doesn't accidentally select labels; the main block gets tighter
# vertical padding so the schematic fits a normal screen without scrolling.
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    user-select: none;
    -webkit-user-select: none;
}
/* Trim the empty band below the last sidebar control (Run Optimizer
   button) so the sidebar fits a typical viewport without scrolling. */
section[data-testid="stSidebar"] > div:last-child,
[data-testid="stSidebarUserContent"] {
    padding-bottom: 0.5rem !important;
}
/* Disabled sidebar buttons (Run Optimizer when the slider state already
   matches the cached res) should not change the cursor on hover -
   Streamlit's default `cursor: not-allowed` reads like an error state,
   but here the disabled-ness is a passive "no work to do" signal paired
   with the red Play button on the chart. Leave the cursor as the
   default arrow so the only feedback is visual (greyed text + red Play). */
section[data-testid="stSidebar"] .stButton > button:disabled,
section[data-testid="stSidebar"] .stButton > button:disabled:hover {
    cursor: default !important;
}
/* Home-link logo at the very top of the sidebar, in normal document flow
   so it scrolls with the sidebar content (not pinned to the viewport).
   The sidebarless Knapsack and Diet apps still use a position:fixed
   variant of this same class: those have no sidebar to anchor to. */
.home-logo-corner {
    display: inline-block;   /* shrink to the icon so only the G is clickable */
    margin: 0 0 0.75rem;
}
.home-logo-corner img {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: block;
}
/* Hide Streamlit's sticky sidebar header (which hosts the «« collapse
   arrow) so the home-logo sits at the very top of the sidebar with no
   chrome above it. Trade-off: the user can no longer collapse the sidebar
   via the button. The sidebar is the app's control panel and is meant
   to stay visible, so this is fine for this app. */
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
/* Slightly tighten the gap between each sidebar slider and the widget
   below it. Most sliders are followed by another slider's label, so this
   pulls those labels closer to the preceding track. */
section[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-bottom: -0.25rem !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
    padding-bottom: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs: four absolute-height sliders + two action buttons.
# Home link: clicking the Griffith PSE logo navigates back to the portfolio
# site. Same-tab navigation since the user is leaving the demo. Lives at
# the top of the sidebar (the upper-left of the page when expanded), so
# it's visually consistent with the corner-pinned logo on the sidebarless
# Knapsack and Diet apps. Image is embedded from the local favicon.png as
# a base64 data URL: the link still navigates to griffith-pse.com when
# clicked, but loading the page itself doesn't make any third-party request.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.sidebar.markdown(
    f'<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" '
        f'alt="Griffith PSE: home" width="32" height="32" style="width:32px;height:32px;border-radius:4px;display:block" />'
    f'</a>',
    unsafe_allow_html=True,
)

# Inline the unit hint into the section header (rather than a separate
# `st.caption(...)` line below it) so the sidebar stays compact. The span
# styling matches Streamlit's default caption: gray, smaller, regular weight.
st.sidebar.markdown(
    '## Initial Conditions &nbsp; <span style="color: rgba(49, 51, 63, 0.6); '
    'font-size: 0.875rem; font-weight: 400;">tank height (cm)</span>',
    unsafe_allow_html=True,
)

# Slider ranges derived from model variable bounds converted to absolute height
x1init = st.sidebar.slider("x₁", 7.5, 28.0, 19.0, 0.1, format="%.1f", key="x1init")
x2init = st.sidebar.slider("x₂", 7.5, 28.0,  9.0, 0.1, format="%.1f", key="x2init")
x3init = st.sidebar.slider("x₃", 3.5, 28.0, 19.2, 0.1, format="%.1f", key="x3init")
x4init = st.sidebar.slider("x₄", 4.5, 28.0, 16.3, 0.1, format="%.1f", key="x4init")

# Convert absolute heights → deviations for the solver. The Pyomo model
# below uses deviation variables z_i = x_i - x_ss_i throughout.
z1init = x1init - XSS[1]
z2init = x2init - XSS[2]
z3init = x3init - XSS[3]
z4init = x4init - XSS[4]

def _init_ss():
    # `on_click` callback for the "Initialize at Steady State" button.
    # Writing to the slider's session_state key resets its position.
    st.session_state["x1init"] = XSS[1]
    st.session_state["x2init"] = XSS[2]
    st.session_state["x3init"] = XSS[3]
    st.session_state["x4init"] = XSS[4]

st.sidebar.button("Initialize at Steady State", on_click=_init_ss, use_container_width=True)

# Controller parameters: objective weighting + discretization grid.
#   ρ     : control-effort weight in the tracking objective. 0 = pure
#            regulator, larger values smooth the pump trajectories.
#   h, nfe: orthogonal collocation grid: nfe finite elements of length h
#            seconds each. Changes take effect on the next Run Optimizer click.
st.sidebar.header("Controller Parameters")
rho    = st.sidebar.slider("Control penalty, ρ",    0.0, 1.0, 0.0, 0.01, key="rho")
h_step = st.sidebar.slider("Step size, h (s)",      1,    30,  10,  1,   key="h_step")
nfe    = st.sidebar.slider("Finite elements, nfe",  5,    30,  15,  1,   key="nfe")
st.sidebar.caption(f"Total horizon: {h_step * nfe} s")

# Tuple of every input that affects the solver result. Used both to stash
# alongside `res` after a solve and to compare against the current slider
# state below, so the Run Optimizer button can grey itself out when there's
# nothing new to compute. Float equality is reliable here: sliders return
# the same float for the same position, so an unchanged slider yields an
# identical tuple element across reruns.
current_inputs = (x1init, x2init, x3init, x4init, h_step, nfe, rho)
_cached_res = st.session_state.get("res")
has_pending_changes = (
    _cached_res is None or _cached_res.get("inputs") != current_inputs
)

# `solve_btn` is True for the rerun immediately after the button is clicked.
# The actual handler is in the main layout section near the bottom of the file.
# `disabled=not has_pending_changes` greys the button out when the cached
# `res` already reflects the current slider values; the Play button in the
# Simulation tab is highlighted red instead, signalling that replay is the
# available action.
solve_btn = st.sidebar.button(
    "Run Optimizer",
    type="primary",
    use_container_width=True,
    disabled=not has_pending_changes,
)

# ── Solver ────────────────────────────────────────────────────────────────────
#
# `solve_model` builds and solves the Pyomo NLP. The continuous-time tank
# ODEs are discretized with orthogonal collocation: the time horizon is
# split into N finite elements of length h, and each element carries `ncp`
# interior collocation points where the ODEs must be satisfied. State
# values at element boundaries (z_i0[ii]) match the last collocation point
# of the previous element, giving a continuous solution.

# One solve at a time per machine: every Streamlit session runs app.py in
# its own thread of this one process, and the solver-log capture redirects
# process-global stdout. Overlapping captures corrupt each other and fail
# both solves, so every solve serializes behind this lock.
_SOLVE_LOCK = threading.Lock()


def solve_model(zi, nfe, h, rho):
    m = pyo.ConcreteModel()

    # Discretization sizing: `nfe` finite elements of `h` s each, with 3
    # collocation points per element (Radau quadrature). nfe and h are
    # exposed in the sidebar so the user can probe the resolution / horizon
    # trade-off.
    N, ncp = nfe, 3

    # Index sets:
    #   i   = elements 0 .. N-1   (interior of horizon)
    #   ii  = elements 1 .. N     (used for boundary continuity)
    #   iii = boundaries 0 .. N   (one more point than elements)
    #   c   = collocation points 1 .. ncp inside each element
    #   t   = tank index 1 .. 4 (used by per-tank parameters)
    m.i   = pyo.Set(initialize=pyo.RangeSet(0, N-1))
    m.ii  = pyo.Set(initialize=pyo.RangeSet(1, N))
    m.iii = pyo.Set(initialize=pyo.RangeSet(0, N))
    m.c   = pyo.Set(initialize=pyo.RangeSet(1, ncp))
    m.t   = pyo.Set(initialize=pyo.RangeSet(1, 4))

    # State variables. For each tank i in 1..4:
    #   z_i0[ii] : the deviation level at element-boundary ii (unbounded -
    #              continuity with z_i[i-1, ncp] is enforced by constraint,
    #              and the surrounding collocation values are bounded).
    #   z_i[i,c] : the deviation level at collocation point c of element i.
    #              Bounds on these keep absolute heights in physically
    #              meaningful ranges given the steady-state offsets zss.
    #   z_i_dot[i,c] : the time derivative at the same collocation point
    #                  (unbounded: derived from z via the ODE constraints).
    m.z10 = pyo.Var(m.iii)
    m.z20 = pyo.Var(m.iii)
    m.z30 = pyo.Var(m.iii)
    m.z40 = pyo.Var(m.iii)
    m.z1  = pyo.Var(m.i, m.c, bounds=(-6.5, 14))
    m.z2  = pyo.Var(m.i, m.c, bounds=(-6.5, 14))
    m.z3  = pyo.Var(m.i, m.c, bounds=(-10.7, 13.8))
    m.z4  = pyo.Var(m.i, m.c, bounds=(-16.8, 6.7))
    m.z1dot = pyo.Var(m.i, m.c)
    m.z2dot = pyo.Var(m.i, m.c)
    m.z3dot = pyo.Var(m.i, m.c)
    m.z4dot = pyo.Var(m.i, m.c)

    # Control inputs. v_p[i] is the deviation pump flow for pump p over
    # element i (piecewise constant within an element). Bounds map back to
    # absolute pump flows in [0, 60] given the steady-state offsets uss.
    m.v1  = pyo.Var(m.i, bounds=(-43.4, 16.6))
    m.v2  = pyo.Var(m.i, bounds=(-35.4, 24.6))

    # Scalar that holds the tracking objective value (filled by track_con).
    m.track = pyo.Var(within=pyo.NonNegativeReals)

    # Physical / numerical parameters:
    #   smalla = nozzle (outflow) cross-section areas
    #   biga   = tank cross-section areas
    #   xss    = steady-state heights (also defined as XSS at module level)
    #   uss    = steady-state pump flows
    #   g      = gravity (cm/s^2)
    #   gamma  = pump split fraction (0..1) directed to the lower tank
    #   h      = element length in seconds
    m.smalla = pyo.Param(m.t, initialize={1: .233, 2: .242, 3: .127, 4: .127})
    m.biga   = pyo.Param(m.t, initialize={1: 50.27, 2: 50.27, 3: 28.27, 4: 28.27})
    m.xss    = pyo.Param(m.t, initialize={1: 14, 2: 14, 3: 14.2, 4: 21.3})
    m.uss    = pyo.Param(pyo.RangeSet(1, 2), initialize={1: 43.4, 2: 35.4})
    m.g      = pyo.Param(initialize=981)
    m.gamma  = pyo.Param(initialize=.4)
    m.h      = pyo.Param(initialize=h)
    m.rho    = pyo.Param(initialize=rho)
    # Radau collocation matrix: omega[k,c] is the integration weight from
    # collocation point k applied when reconstructing the state at c. With
    # ncp=3 these are the standard Radau-IIA coefficients.
    omega_data = {
        (1,1): 0.19681547722366, (1,2): 0.39442431473909, (1,3): 0.37640306270047,
        (2,1): -0.06553542585020, (2,2): 0.29207341166523, (2,3): 0.51248582618842,
        (3,1): 0.02377097434822,  (3,2): -0.04154875212600, (3,3): 0.11111111111111,
    }
    m.omega  = pyo.Param(m.c, m.c, initialize=omega_data)

    # Initial-condition parameters wired in from the sidebar slider values.
    m.z1init = pyo.Param(initialize=zi[0])
    m.z2init = pyo.Param(initialize=zi[1])
    m.z3init = pyo.Param(initialize=zi[2])
    m.z4init = pyo.Param(initialize=zi[3])

    # ── Tank dynamics ────────────────────────────────────────────────────────
    # Each z_i_dot constraint is a Torricelli-style mass balance for tank i:
    #   inflow from upstream tank's drain (sqrt term)
    #   - own drain (sqrt term)
    #   + share of pump flow directed at this tank.
    # The cross-coupling is structural: pump 1 feeds tanks 1 and 4,
    # pump 2 feeds tanks 2 and 3.
    def z1dot_def(m, i, c):
        return m.z1dot[i,c] == (
            -(m.smalla[1]/m.biga[1])*pyo.sqrt(2*m.g*(m.z1[i,c]+m.xss[1]))
            +(m.smalla[3]/m.biga[1])*pyo.sqrt(2*m.g*(m.z3[i,c]+m.xss[3]))
            +(m.gamma/m.biga[1])*(m.v1[i]+m.uss[1]))
    m.z1dot_con = pyo.Constraint(m.i, m.c, rule=z1dot_def)

    def z2dot_def(m, i, c):
        return m.z2dot[i,c] == (
            -(m.smalla[2]/m.biga[2])*pyo.sqrt(2*m.g*(m.z2[i,c]+m.xss[2]))
            +(m.smalla[4]/m.biga[2])*pyo.sqrt(2*m.g*(m.z4[i,c]+m.xss[4]))
            +(m.gamma/m.biga[2])*(m.v2[i]+m.uss[2]))
    m.z2dot_con = pyo.Constraint(m.i, m.c, rule=z2dot_def)

    def z3dot_def(m, i, c):
        return m.z3dot[i,c] == (
            -(m.smalla[3]/m.biga[3])*pyo.sqrt(2*m.g*(m.z3[i,c]+m.xss[3]))
            +((1-m.gamma)/m.biga[3])*(m.v2[i]+m.uss[2]))
    m.z3dot_con = pyo.Constraint(m.i, m.c, rule=z3dot_def)

    def z4dot_def(m, i, c):
        return m.z4dot[i,c] == (
            -(m.smalla[4]/m.biga[4])*pyo.sqrt(2*m.g*(m.z4[i,c]+m.xss[4]))
            +((1-m.gamma)/m.biga[4])*(m.v1[i]+m.uss[1]))
    m.z4dot_con = pyo.Constraint(m.i, m.c, rule=z4dot_def)

    # ── Collocation (state at collocation points) ────────────────────────────
    # Each z_i[i,c] equals the start-of-element value plus the integral of
    # z_i_dot from element start up to point c, approximated with the omega
    # weights. This is the "Lagrange" (or implicit Runge-Kutta) form.
    def z1_def(m, i, c):
        return m.z1[i,c] == m.z10[i] + m.h*sum(m.omega[k,c]*m.z1dot[i,k] for k in m.c)
    m.z1_con = pyo.Constraint(m.i, m.c, rule=z1_def)

    def z2_def(m, i, c):
        return m.z2[i,c] == m.z20[i] + m.h*sum(m.omega[k,c]*m.z2dot[i,k] for k in m.c)
    m.z2_con = pyo.Constraint(m.i, m.c, rule=z2_def)

    def z3_def(m, i, c):
        return m.z3[i,c] == m.z30[i] + m.h*sum(m.omega[k,c]*m.z3dot[i,k] for k in m.c)
    m.z3_con = pyo.Constraint(m.i, m.c, rule=z3_def)

    def z4_def(m, i, c):
        return m.z4[i,c] == m.z40[i] + m.h*sum(m.omega[k,c]*m.z4dot[i,k] for k in m.c)
    m.z4_con = pyo.Constraint(m.i, m.c, rule=z4_def)

    # ── Boundary continuity ──────────────────────────────────────────────────
    # The start of element ii equals the last collocation point of element
    # ii-1. This stitches consecutive elements into a single trajectory.
    m.z10_con = pyo.Constraint(m.ii, rule=lambda m, ii: m.z10[ii] == m.z1[ii-1, ncp])
    m.z20_con = pyo.Constraint(m.ii, rule=lambda m, ii: m.z20[ii] == m.z2[ii-1, ncp])
    m.z30_con = pyo.Constraint(m.ii, rule=lambda m, ii: m.z30[ii] == m.z3[ii-1, ncp])
    m.z40_con = pyo.Constraint(m.ii, rule=lambda m, ii: m.z40[ii] == m.z4[ii-1, ncp])

    # ── Initial conditions ───────────────────────────────────────────────────
    # Pin the t=0 boundary values to the parameters wired in from the sidebar.
    m.z1init_con = pyo.Constraint(expr=m.z10[0] == m.z1init)
    m.z2init_con = pyo.Constraint(expr=m.z20[0] == m.z2init)
    m.z3init_con = pyo.Constraint(expr=m.z30[0] == m.z3init)
    m.z4init_con = pyo.Constraint(expr=m.z40[0] == m.z4init)

    # ── Objective ────────────────────────────────────────────────────────────
    # Minimize sum of squared state deviations (drive every tank back to
    # steady state) plus ρ·sum of squared pump-input deviations (penalize
    # control effort). State term is summed over element boundaries; control
    # term is summed over elements (pump inputs are piecewise constant
    # within an element). Wrapped in a defining constraint plus a linear
    # min-track objective so the solver sees a clean linear objective with
    # a nonlinear constraint, which often converges better than an inline
    # quadratic objective for problems of this shape.
    m.track_con = pyo.Constraint(
        expr=m.track == sum(m.z10[i]**2 + m.z20[i]**2 + m.z30[i]**2 + m.z40[i]**2 for i in m.iii)
                      + m.rho * sum(m.v1[i]**2 + m.v2[i]**2 for i in m.i))
    m.obj = pyo.Objective(expr=m.track, sense=pyo.minimize)

    # ── Solve ────────────────────────────────────────────────────────────────
    # pounce's binary is bundled in the pyomo-pounce wheel, so SolverFactory
    # finds it without any path lookup. `tee=True` streams solver output to
    # stdout, which we redirect into a StringIO so it can be displayed in
    # the Logs tab.
    solver = pyo.SolverFactory('pounce')
    buf = io.StringIO()
    with _SOLVE_LOCK, contextlib.redirect_stdout(buf):
        result = solver.solve(m, tee=True)
    status = str(result.solver.termination_condition)

    # Extract everything the UI needs into plain Python lists. Times are in
    # seconds (each element is h seconds long (user-set via the h_step
    # slider), so element index k → t = h·k).
    t_pts = list(m.iii)
    return {
        "status": status,
        "log": buf.getvalue(),
        "h": h,
        "t": [k * h for k in t_pts],
        "z10": [pyo.value(m.z10[i]) for i in t_pts],
        "z20": [pyo.value(m.z20[i]) for i in t_pts],
        "z30": [pyo.value(m.z30[i]) for i in t_pts],
        "z40": [pyo.value(m.z40[i]) for i in t_pts],
        "v1": [pyo.value(m.v1[i]) for i in m.i],
        "v2": [pyo.value(m.v2[i]) for i in m.i],
    }


# ── Animated schematic ────────────────────────────────────────────────────────
#
# `build_tank_figure` returns a plotly Figure with animation frames: one
# per element boundary in the optimization horizon. The figure has three
# layers built up over the function:
#   1. Static `shapes`: tanks, walls, pipes, valves, pumps, gauges.
#   2. Animated `traces`: water fill in each tank, flow streams, pump
#      gauge fills, water-level labels. Rebuilt once per frame.
#   3. Layout: Play/Pause buttons + frame slider.
# The function is long because every shape is positioned by hand; once the
# coordinate system is set up, each chunk is independent and additive.

def build_tank_figure(res):
    """Animated schematic matching the quad-tank physical layout."""
    import math as _m

    # ── coordinate layout ────────────────────────────────────────────────────
    # Compact y-coordinates so the data aspect ratio suits a normal screen.
    # x range ≈ 12 units wide, y range ≈ 7.4 units tall → ~0.62 aspect ratio.
    # Tank boundaries: (x_left, y_bottom, x_right, y_top)
    TB = {
        1: (1.2, 1.5, 4.8, 3.7),   # bottom-left  (large): raised for visible drain gap
        2: (5.2, 1.5, 8.8, 3.7),   # bottom-right (large): raised for visible drain gap
        3: (1.2, 4.6, 4.8, 6.4),   # top-left     (small)
        4: (5.2, 4.6, 8.8, 6.4),   # top-right    (small)
    }

    DISP_MAX  = 30.0   # cm: full-scale height for display
    PW        = 0.13   # pipe half-width
    WALL      = 0.12   # tank wall thickness
    TOP_Y_U1  = 7.15   # u1 overhead pipe centre y  (u1 left → T4 right, higher)
    TOP_Y_U2  = 6.68   # u2 overhead pipe centre y  (u2 right → T3 left, lower)
    GAMMA_Y   = 3.95   # γ valve centre y  (in the gap between upper/lower tanks)
    RES_TOP   = 0.82   # top of reservoir

    # Outer (pump) pipe x positions
    LP = 0.42;  RP = 9.58

    # Left side: γ feed on the LEFT (close to pump), drain on the RIGHT
    # so the γ₁ horizontal branch (LP → LX_P) never crosses the drain pipe.
    LX_P = 1.8   # pump-direct: γ₁ fraction of u₁ ↓ Tank 1  (left side of tank)
    LX_D = 3.2   # drain pipe : Tank 3 ↓ Tank 1, top-feed ↓ Tank 3 (right side)

    # Right side: mirror: γ feed on the RIGHT (close to pump), drain on the LEFT
    RX_P = 8.2   # pump-direct: γ₂ fraction of u₂ ↓ Tank 2  (right side of tank)
    RX_D = 6.8   # drain pipe : Tank 4 ↓ Tank 2, top-feed ↓ Tank 4 (left side)

    # ── physics constants ────────────────────────────────────────────────────
    _G    = 981
    _SA   = {1: .233, 2: .242, 3: .127, 4: .127}  # nozzle areas
    _GAMA = 0.4
    _USS  = {1: 43.4, 2: 35.4}

    # ── pump gauge constants ─────────────────────────────────────────────────
    # Gauge bars sit to the outside of the diagram (left of u₁, right of u₂).
    # u_actual = v + USS;  max possible = USS + upper_v_bound
    _U_MAX  = {1: 60.0, 2: 60.0}   # v1 ≤ 16.6 → u1 ≤ 60; v2 ≤ 24.6 → u2 ≤ 60
    GX  = {1: (-0.95, -0.45),       # (x_left, x_right) for u₁ gauge
           2: (10.45,  10.95)}      # mirrored, width 0.50
    GH  = 1.80                      # gauge height (in data units)
    # Centre gauge vertically around the pump circle mid-point
    # PY = (RES_TOP + GAMMA_Y) / 2 = (0.82 + 3.95) / 2 = 2.385
    GY0 = (RES_TOP + GAMMA_Y) / 2 - GH / 2   # ≈ 1.485

    # ── compute flow rates at each time step ─────────────────────────────────
    # All flows in ml/s: same units, single normaliser:
    #   drain_i  = SA_i * sqrt(2g*h_i)         [ml/s, Torricelli]
    #   pump_*   = fraction * (v + uss)         [ml/s]

    t_pts = res["t"]
    n_pts = len(t_pts)

    # Convert deviation results back to absolute heights for display.
    actual = [
        {1: res["z10"][k] + XSS[1],
         2: res["z20"][k] + XSS[2],
         3: res["z30"][k] + XSS[3],
         4: res["z40"][k] + XSS[4]}
        for k in range(n_pts)
    ]

    drain   = {i: [] for i in range(1, 5)}   # Torricelli outflows
    p_dir   = {i: [] for i in (1, 2)}        # γ × u → Tank 1/2 directly
    p_top   = {i: [] for i in (3, 4)}        # (1-γ) × u → Tank 3/4 via top pipe

    for k in range(n_pts):
        vi  = min(k, len(res["v1"]) - 1)
        h   = {i: actual[k][i] for i in range(1, 5)}
        v1k = res["v1"][vi];  v2k = res["v2"][vi]

        for i in range(1, 5):
            drain[i].append(_SA[i] * _m.sqrt(max(0.0, 2 * _G * h[i])))

        # pump direct: γ fraction from each pump into the paired lower tank
        # u1 (left pump)  → γ×u1 into Tank 1
        # u2 (right pump) → γ×u2 into Tank 2
        p_dir[1].append(max(0.0, _GAMA * (v1k + _USS[1])))
        p_dir[2].append(max(0.0, _GAMA * (v2k + _USS[2])))

        # pump top: (1-γ) fraction via top bar into upper tanks
        # u1 → (1-γ)×u1 into Tank 4   (right upper)
        # u2 → (1-γ)×u2 into Tank 3   (left upper)
        p_top[4].append(max(0.0, (1 - _GAMA) * (v1k + _USS[1])))
        p_top[3].append(max(0.0, (1 - _GAMA) * (v2k + _USS[2])))

    max_flow = max(
        max(v for d in drain.values() for v in d),
        max(v for d in p_dir.values() for v in d),
        max(v for d in p_top.values() for v in d),
    ) or 1.0

    # ── flow stream definitions ───────────────────────────────────────────────
    # All water is the same blue; width encodes flow magnitude.
    C_WATER = "rgb(28, 108, 215)"

    # SEGS: (flow_dict, key, pipe_x, y_high, y_low, colour, normaliser)
    SEGS = [
        # Torricelli drains: water starts at bottom of gray nozzle pipe
        (drain,  1, LX_D, TB[1][1] - 0.50*(TB[1][1]-RES_TOP), 0.0, C_WATER, max_flow),
        (drain,  2, RX_D, TB[2][1] - 0.50*(TB[2][1]-RES_TOP), 0.0, C_WATER, max_flow),
        (drain,  3, LX_D, TB[1][3],  TB[1][1],  C_WATER, max_flow),  # T3 → T1
        (drain,  4, RX_D, TB[2][3],  TB[2][1],  C_WATER, max_flow),  # T4 → T2
        # γ direct pump feeds: bottom of feed pipe = top of lower tank
        (p_dir,  1, LX_P, TB[1][3],  TB[1][1],  C_WATER, max_flow),  # γ×u1 → T1
        (p_dir,  2, RX_P, TB[2][3],  TB[2][1],  C_WATER, max_flow),  # γ×u2 → T2
        # (1-γ) overhead feeds: bottom of feed pipe = top of upper tank
        (p_top,  3, LX_D, TB[3][3],  TB[3][1],  C_WATER, max_flow),  # (1-γ)×u2 → T3
        (p_top,  4, RX_D, TB[4][3],  TB[4][1],  C_WATER, max_flow),  # (1-γ)×u1 → T4
    ]

    # ── helper ───────────────────────────────────────────────────────────────
    # `water_top_y` maps a tank height in cm to the vertical pixel position
    # of the water surface inside the schematic. `stream_rect` builds a
    # rectangular flow stream as a filled scatter polygon. `make_traces`
    # produces all dynamic shapes for a single animation frame.
    def water_top_y(tk, h):
        x0, y0, x1, y1 = TB[tk]
        return y0 + min(1.0, max(0.01, h / DISP_MAX)) * (y1 - y0)

    def stream_rect(xp, y_top, y_bot, hw, color):
        return go.Scatter(
            x=[xp-hw, xp-hw, xp+hw, xp+hw, xp-hw],
            y=[y_bot,  y_top,  y_top,  y_bot,  y_bot],
            fill="toself", fillcolor=color,
            mode="none",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )

    def make_traces(heights, step):
        traces = []

        # Water fill in each tank: same blue for all
        for tk, col in [(1, "rgb(28, 108, 215)"),
                        (2, "rgb(28, 108, 215)"),
                        (3, "rgb(28, 108, 215)"),
                        (4, "rgb(28, 108, 215)")]:
            x0, y0, x1, _ = TB[tk]
            wy = water_top_y(tk, heights[tk])
            traces.append(go.Scatter(
                x=[x0 - WALL, x0 - WALL, x1 + WALL, x1 + WALL, x0 - WALL],
                y=[y0 - 0.02, wy,        wy,        y0 - 0.02, y0 - 0.02],
                fill="toself", fillcolor=col, mode="none", line=dict(width=0),
                hovertemplate=f"Tank {tk}: {heights[tk]:.1f} cm<extra></extra>",
                showlegend=False,
            ))

        # Flow streams: width proportional to physical flow rate
        for fd, key, xp, y_top, y_bot, col, norm in SEGS:
            hw = PW * (0.12 + 0.88 * min(1.0, fd[key][step] / norm))
            traces.append(stream_rect(xp, y_top, y_bot, hw, col))

        # ── pump gauge fills (animated) ──────────────────────────────────────
        vi = min(step, len(res["v1"]) - 1)
        u_actual = {1: res["v1"][vi] + _USS[1],
                    2: res["v2"][vi] + _USS[2]}
        for pump in (1, 2):
            gx0, gx1 = GX[pump]
            frac = max(0.0, min(1.0, u_actual[pump] / _U_MAX[pump]))
            gy1  = GY0 + frac * GH
            # filled bar
            traces.append(go.Scatter(
                x=[gx0, gx0, gx1, gx1, gx0],
                y=[GY0,  gy1,  gy1,  GY0,  GY0],
                fill="toself", fillcolor="rgb(28, 108, 215)",
                mode="none", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            # current-value label at top of bar
            traces.append(go.Scatter(
                x=[(gx0 + gx1) / 2],
                y=[max(gy1 + 0.12, GY0 + 0.22)],
                mode="text",
                text=[f"{u_actual[pump]:.1f}"],
                textfont=dict(size=14, color="#0d0d3a"),
                showlegend=False, hoverinfo="skip",
            ))

        # Water-level labels: follow water surface, shifted toward diagram centre
        # to avoid internal pipes (LX_D=3.2 left, RX_D=6.8 right).
        traces.append(go.Scatter(
            x=[4.0, 6.0, 4.0, 6.0],   # inner half of each tank (toward centre x=5)
            y=[water_top_y(k, heights[k]) + 0.12 for k in range(1, 5)],
            mode="text",
            text=[f"x₁={heights[1]:.1f}",
                  f"x₂={heights[2]:.1f}",
                  f"x₃={heights[3]:.1f}",
                  f"x₄={heights[4]:.1f}"],
            textfont=dict(size=14, color="#0d0d3a"),
            showlegend=False, hoverinfo="skip",
        ))
        return traces

    # One animation frame per time step. Plotly cycles through these when
    # the user clicks Play; the slider also exposes them individually.
    frames = [go.Frame(name=str(k), data=make_traces(actual[k], k))
              for k in range(n_pts)]

    # ── static shapes ─────────────────────────────────────────────────────────
    # `shapes` is a single flat list of plotly shape dicts: built up
    # additively below. Order matters only for things that overlap (e.g.
    # the masking rectangle behind the gamma-valve symbols).
    PC = "#6b6b6b"   # unified gray for all structural elements (pipes + tank walls)

    def pipe(x0, y0, x1, y1, color=PC):
        return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                    fillcolor=color, line=dict(width=0), layer="below")

    shapes = []

    # Reservoir
    shapes.append(dict(type="rect", x0=0.0, y0=0.0, x1=10.0, y1=RES_TOP,
                       fillcolor="rgb(28, 108, 215)", line=dict(color="#4a7a9b", width=2)))

    # Tank walls (open-top U-shape: left, right, bottom slabs)
    for tk, (x0, y0, x1, y1) in TB.items():
        wc = PC
        shapes += [
            dict(type="rect", x0=x0-WALL, y0=y0-WALL, x1=x0,       y1=y1,  fillcolor=wc, line=dict(width=0)),
            dict(type="rect", x0=x1,       y0=y0-WALL, x1=x1+WALL,  y1=y1,  fillcolor=wc, line=dict(width=0)),
            dict(type="rect", x0=x0-WALL, y0=y0-WALL, x1=x1+WALL,  y1=y0,  fillcolor=wc, line=dict(width=0)),
        ]

    # ── pipes (layer=below) ───────────────────────────────────────────────────

    # Left outer pipe: u1 rises to TOP_Y_U1 (higher overhead)
    shapes.append(pipe(LP-PW, RES_TOP, LP+PW, TOP_Y_U1+PW))
    # Right outer pipe: u2 rises to TOP_Y_U2 (lower overhead)
    shapes.append(pipe(RP-PW, RES_TOP, RP+PW, TOP_Y_U2+PW))

    # Overhead pipe A: u1 (left pump) → Tank 4 (right upper): higher pipe
    shapes.append(pipe(LP-PW,   TOP_Y_U1-PW, RX_D+PW, TOP_Y_U1+PW))   # horizontal LP→RX_D
    shapes.append(pipe(RX_D-PW, TB[4][3]-0.02, RX_D+PW, TOP_Y_U1+PW)) # drop into T4 (overlap into horizontal to close gap)

    # Overhead pipe B: u2 (right pump) → Tank 3 (left upper): lower pipe
    shapes.append(pipe(LX_D-PW, TOP_Y_U2-PW, RP+PW,   TOP_Y_U2+PW))   # horizontal LX_D→RP
    shapes.append(pipe(LX_D-PW, TB[3][3]-0.02, LX_D+PW, TOP_Y_U2+PW)) # drop into T3 (overlap into horizontal to close gap)

    # Inter-tank drain pipes (between upper and lower tanks)
    shapes.append(pipe(LX_D-PW, TB[1][3]-0.02, LX_D+PW, TB[3][1]+0.02))  # T3 → T1
    shapes.append(pipe(RX_D-PW, TB[2][3]-0.02, RX_D+PW, TB[4][1]+0.02))  # T4 → T2

    # Tank 1/2 bottom drain pipes: shortened to ~25% so water stream is visible below
    _drain_bot = TB[1][1] - 0.50 * (TB[1][1] - RES_TOP)   # ≈ 1.16
    shapes.append(pipe(LX_D-PW, _drain_bot, LX_D+PW, TB[1][1]+0.02))
    shapes.append(pipe(RX_D-PW, _drain_bot, RX_D+PW, TB[2][1]+0.02))

    # γ₁ branch: LP → LX_P (branch stays left of drain pipe, no crossing)
    shapes.append(pipe(LP+PW,   GAMMA_Y-PW, LX_P+PW, GAMMA_Y+PW))   # horizontal
    shapes.append(pipe(LX_P-PW, TB[1][3]-0.02, LX_P+PW, GAMMA_Y+PW))  # vertical drop

    # γ₂ branch: RX_P → RP (branch stays right of drain pipe, no crossing)
    shapes.append(pipe(RX_P-PW, GAMMA_Y-PW, RP-PW,   GAMMA_Y+PW))   # horizontal
    shapes.append(pipe(RX_P-PW, TB[2][3]-0.02, RX_P+PW, GAMMA_Y+PW))  # vertical drop

    # γ valve symbols: P&ID control valve: 4-armed cross at (cx, cy)
    #   Outer arm : D-shaped semicircle (curves LEFT for γ₁, RIGHT for γ₂)
    #   Inner arm : hollow triangle  (◀ for γ₁, ▶ for γ₂)
    #   Top arm   : hollow triangle  ▽ (base at top,    tip at centre)
    #   Bottom arm: hollow triangle  △ (base at bottom, tip at centre)
    VW = 0.36        # arm length  centre → tip / base
    VH = 0.20        # arm half-width = semicircle radius (skinnier triangles)
    BG = "#f8fafd"   # hollow fill matches plot background
    LS = dict(color="#1a1a1a", width=1.8)

    for idx, vx in enumerate([LP, RP]):
        cx, cy = vx, GAMMA_Y

        # Background mask: hides the grey pipe in the gaps between triangle arms.
        # Drawn first (below triangles in the above-layer render order) so that
        # pipes appear to enter the symbol and stop at its edge.
        # Mask sized to exactly the arm length: pipe is visible right up to the
        # triangle edges but hidden in the gaps between arms.
        shapes.append(dict(type="rect",
            x0=cx-VW, y0=cy-VW, x1=cx+VW, y1=cy+VW,
            fillcolor=BG, line=dict(width=0)))

        # Outer arm: stem line + D-shaped semicircle at the end.
        # The stem runs from the triangle intersection (cx,cy) outward,
        # and the semicircle sits at the far end of the stem.
        K  = 0.5523   # bezier quarter-circle approximation constant
        r  = VH       # semicircle radius
        if idx == 0:  # γ₁: stem goes LEFT, semicircle curves LEFT
            ex = cx - VW              # semicircle centre x
            sc = (f"M {ex:.4f} {cy+r:.4f} "
                  f"C {ex-r*K:.4f} {cy+r:.4f} {ex-r:.4f} {cy+r*K:.4f} {ex-r:.4f} {cy:.4f} "
                  f"C {ex-r:.4f} {cy-r*K:.4f} {ex-r*K:.4f} {cy-r:.4f} {ex:.4f} {cy-r:.4f} Z")
        else:         # γ₂: stem goes RIGHT, semicircle curves RIGHT
            ex = cx + VW              # semicircle centre x
            sc = (f"M {ex:.4f} {cy+r:.4f} "
                  f"C {ex+r*K:.4f} {cy+r:.4f} {ex+r:.4f} {cy+r*K:.4f} {ex+r:.4f} {cy:.4f} "
                  f"C {ex+r:.4f} {cy-r*K:.4f} {ex+r*K:.4f} {cy-r:.4f} {ex:.4f} {cy-r:.4f} Z")
        # Stem line: intersection → semicircle flat edge
        shapes.append(dict(type="line",
            x0=cx, y0=cy, x1=ex, y1=cy, line=LS))
        shapes.append(dict(type="path", path=sc, fillcolor=BG, line=LS))

        # Inner arm: hollow triangle (base on inner side, tip at centre)
        bx = cx + VW if idx == 0 else cx - VW   # base x
        shapes.append(dict(type="path",
            path=(f"M {bx:.4f} {cy+VH:.4f} "
                  f"L {cx:.4f} {cy:.4f} "
                  f"L {bx:.4f} {cy-VH:.4f} Z"),
            fillcolor=BG, line=LS))

        # Top arm: hollow triangle ▽
        shapes.append(dict(type="path",
            path=(f"M {cx-VH:.4f} {cy+VW:.4f} "
                  f"L {cx:.4f} {cy:.4f} "
                  f"L {cx+VH:.4f} {cy+VW:.4f} Z"),
            fillcolor=BG, line=LS))

        # Bottom arm: hollow triangle △
        shapes.append(dict(type="path",
            path=(f"M {cx-VH:.4f} {cy-VW:.4f} "
                  f"L {cx:.4f} {cy:.4f} "
                  f"L {cx+VH:.4f} {cy-VW:.4f} Z"),
            fillcolor=BG, line=LS))

    # Pump circles: midway up the pipe from reservoir to valve junction
    PR = 0.22                              # circle radius (smaller than before)
    PY = (RES_TOP + GAMMA_Y) / 2          # halfway between reservoir top and γ valve
    TR = PR                                # circumradius = PR → vertices on circle edge
    for px in [LP, RP]:
        # Circle body (white fill hides the grey pipe behind it)
        shapes.append(dict(type="circle",
                           x0=px-PR, y0=PY-PR, x1=px+PR, y1=PY+PR,
                           fillcolor="white", line=dict(color="#333", width=2)))
        # Hollow triangle inscribed in circle, pointing UP
        shapes.append(dict(type="path",
            path=(f"M {px:.4f} {PY+TR:.4f} "
                  f"L {px+TR*0.866:.4f} {PY-TR*0.5:.4f} "
                  f"L {px-TR*0.866:.4f} {PY-TR*0.5:.4f} Z"),
            fillcolor="white", line=dict(color="#333", width=1.5)))

    # ── pump gauge outlines (static) ─────────────────────────────────────────
    for pump in (1, 2):
        gx0, gx1 = GX[pump]
        # gauge body background
        shapes.append(dict(type="rect", x0=gx0, y0=GY0, x1=gx1, y1=GY0+GH,
                           fillcolor="rgba(210, 225, 245, 0.55)",
                           line=dict(color="#555", width=1.5)))
        # steady-state dashed red line
        ss_y = GY0 + (_USS[pump] / _U_MAX[pump]) * GH
        shapes.append(dict(type="line", x0=gx0, y0=ss_y, x1=gx1, y1=ss_y,
                           line=dict(color="rgba(200, 40, 40, 0.65)", width=1.5, dash="dot")))

    # Steady-state dashed lines inside each tank
    for tk, ss in XSS.items():
        x0, y0, x1, y1 = TB[tk]
        ss_y = y0 + (ss / DISP_MAX) * (y1 - y0)
        shapes.append(dict(type="line", x0=x0, y0=ss_y, x1=x1, y1=ss_y,
                           line=dict(color="rgba(200,40,40,0.55)", width=1.5, dash="dot")))

    # ── annotations ───────────────────────────────────────────────────────────
    annotations = [
        dict(x=(TB[k][0]+TB[k][2])/2, y=TB[k][1]+0.18,
             text=f"<b>Tank {k}</b>", showarrow=False,
             font=dict(size=13, color="#0a0a4e"))
        for k in range(1, 5)
    ] + [
        dict(x=5.0, y=0.35, text="<b>Reservoir</b>", showarrow=False,
             font=dict(size=13, color="#0a0a4e")),
        # pump gauge labels: title above, scale endpoints alongside
        # u labels: midpoint between pump centre and gauge centre, at pump height
        dict(x=(LP + (GX[1][0]+GX[1][1])/2) / 2,
             y=(RES_TOP + GAMMA_Y) / 2,
             text="<b>u₁</b>", showarrow=False, font=dict(size=13, color="#1255a0")),
        dict(x=(RP + (GX[2][0]+GX[2][1])/2) / 2,
             y=(RES_TOP + GAMMA_Y) / 2,
             text="<b>u₂</b>", showarrow=False, font=dict(size=13, color="#1255a0")),
        dict(x=GX[1][0]-0.06, y=GY0+GH, text="60", showarrow=False,
             font=dict(size=14, color="#0d0d3a"), xanchor="right"),
        dict(x=GX[1][0]-0.06, y=GY0,    text="0",  showarrow=False,
             font=dict(size=14, color="#0d0d3a"), xanchor="right"),
        dict(x=GX[2][1]+0.06, y=GY0+GH, text="60", showarrow=False,
             font=dict(size=14, color="#0d0d3a"), xanchor="left"),
        dict(x=GX[2][1]+0.06, y=GY0,    text="0",  showarrow=False,
             font=dict(size=14, color="#0d0d3a"), xanchor="left"),
        dict(x=LP-VW-VH-0.12, y=GAMMA_Y, text="γ₁=0.4", showarrow=False,
             font=dict(size=13, color="#8b0000"), xanchor="right"),
        dict(x=RP+VW+VH+0.12, y=GAMMA_Y, text="γ₂=0.4", showarrow=False,
             font=dict(size=13, color="#8b0000"), xanchor="left"),
    ]

    # Slider step list: one step per frame, scrubbing immediately to that
    # frame without playback. The label is the absolute time in seconds.
    slider_steps = [
        {"args": [[str(k)], {"frame": {"duration": 0}, "mode": "immediate",
                              "transition": {"duration": 0}}],
         "label": f"{t}s", "method": "animate"}
        for k, t in enumerate(t_pts)
    ]

    # Assemble the figure. Initial `data` is the first frame's traces; the
    # `frames` argument is consumed by the Play button and slider. Layout
    # disables both axes (purely positional drawing) and locks aspect ratio
    # so the schematic doesn't squish.
    return go.Figure(
        data=make_traces(actual[0], 0),
        frames=frames,
        layout=go.Layout(
            xaxis=dict(visible=False, range=[-1.0, 11.0]),
            yaxis=dict(visible=False, range=[-0.1, 7.8],
                       scaleanchor="x", scaleratio=1),
            shapes=shapes,
            annotations=annotations,
            height=600,
            margin=dict(t=10, b=90, l=15, r=15),
            plot_bgcolor="#f8fafd",
            paper_bgcolor="#f8fafd",
            updatemenus=[{
                "type": "buttons",
                "direction": "right",
                "x": 0.0, "xanchor": "left", "y": 0.0, "yanchor": "top",
                "buttons": [
                    {"label": "▶  Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 400, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 150}}]},
                    {"label": "⏸  Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "active": 0,
                "steps": slider_steps,
                "x": 0.0, "len": 1.0,
                "currentvalue": {"prefix": "Time: ", "suffix": " s",
                                  "font": {"size": 13}, "xanchor": "center"},
                "pad": {"t": 40, "b": 10},
                "transition": {"duration": 0},
            }],
        ),
    )


# ── Time-series plots ─────────────────────────────────────────────────────────
#
# Compact alternative view: two stacked subplots (tank-level deviations on
# top, pump-input deviations on bottom). Useful for reading the full
# trajectory at a glance instead of frame-by-frame.

def build_timeseries(res):
    # Two time grids: `t` is element boundaries (where states are defined),
    # `ti` is element starts (where pump inputs are defined: they are
    # piecewise-constant within an element). Element width is h seconds
    # (user-set via the h_step slider), so element index k → t = h·k.
    t  = res["t"]
    # Pump inputs are piecewise-constant per element: N values over N+1
    # element boundaries. Repeat the last value at the final boundary so a
    # step ('hv') trace holds it across the last element.
    ti = [k * res["h"] for k in range(len(res["v1"]) + 1)]

    # Two-row subplot grid; legends are split by row using `legend2`.
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Tank Levels (deviation from steady-state)", "Pump Inputs"),
        vertical_spacing=0.18,
    )

    for key, color, label in [
        ("z10", "#1565C0", "Tank 1  (x₁)"),
        ("z20", "#E53935", "Tank 2  (x₂)"),
        ("z30", "#0288D1", "Tank 3  (x₃)"),
        ("z40", "#F57C00", "Tank 4  (x₄)"),
    ]:
        fig.add_trace(go.Scatter(x=t, y=res[key], name=label,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"{label}: %{{y:.2f}} cm<extra></extra>"),
                      row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    for data, color, label in [
        (res["v1"], "#6A1B9A", "Pump 1 (u₁)"),
        (res["v2"], "#2E7D32", "Pump 2 (u₂)"),
    ]:
        fig.add_trace(go.Scatter(x=ti, y=data + data[-1:], name=label,
                                 line=dict(color=color, width=2, shape="hv"),
                                 mode="lines",
                                 legend="legend2"),
                      row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Deviation (cm)", row=1, col=1)
    fig.update_yaxes(title_text="Flow rate (ml/s)", row=2, col=1)
    fig.update_layout(
        height=560, margin=dict(t=50, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.04, x=0, xanchor="left"),
        legend2=dict(orientation="h", y=0.42, x=0, xanchor="left",
                     yanchor="bottom"),
    )
    return fig


# ── Formulation tab ───────────────────────────────────────────────────────────
#
# Static content describing the model and the optimization problem. Equations
# render via Streamlit's built-in KaTeX (`$$...$$` in markdown). The parameter
# table uses HTML for tighter spacing: Streamlit's KaTeX does NOT process
# `$...$` inside `unsafe_allow_html=True` blocks, so symbol cells use Unicode
# subscripts (Aᵢ, aᵢ, …) rather than LaTeX.
#
# References at the bottom point to the original four-tank-system paper for
# the model and to Biegler's textbook for the simultaneous direct-transcription
# approach used to discretize the continuous-time OCP.

def render_formulation_tab():
    st.markdown(r"""
### Process model

The four-tank system [Raff et al., 2006] models water levels in four
interconnected tanks driven by two pumps with a flow-split valve at each
pump outlet. Pump $k$ sends a fraction $\gamma_k$ of its flow to its
paired lower tank and the remaining $1-\gamma_k$ overhead to the
diagonally-opposite upper tank, which then drains into the *other* lower
tank. Mass balance gives four nonlinear ODEs:

$$\dot{x}_1 = -\tfrac{a_1}{A_1}\sqrt{2 g x_1} + \tfrac{a_3}{A_1}\sqrt{2 g x_3} + \tfrac{\gamma_1}{A_1}u_1$$

$$\dot{x}_2 = -\tfrac{a_2}{A_2}\sqrt{2 g x_2} + \tfrac{a_4}{A_2}\sqrt{2 g x_4} + \tfrac{\gamma_2}{A_2}u_2$$

$$\dot{x}_3 = -\tfrac{a_3}{A_3}\sqrt{2 g x_3} + \tfrac{1-\gamma_2}{A_3}u_2$$

$$\dot{x}_4 = -\tfrac{a_4}{A_4}\sqrt{2 g x_4} + \tfrac{1-\gamma_1}{A_4}u_1$$

with $x_i$ the level in tank $i$ (cm), $u_k$ the pump-$k$ flow rate (ml/s),
$A_i$ the tank cross-section (cm²), $a_i$ the outlet area (cm²),
$\gamma_k$ the flow-split ratio, and $g$ gravitational acceleration
(cm/s²).
""")

    st.markdown("""
<div style="margin: 0.75rem 0 1.25rem 0;">
<table style="border-collapse: collapse; font-size: 0.95rem; margin-bottom: 0.6rem;">
  <thead>
    <tr style="border-bottom: 1px solid #dee2e6;">
      <th style="padding: 0.4rem 0.9rem; text-align: left; font-weight: 600;">Parameter</th>
      <th style="padding: 0.4rem 0.9rem; text-align: right;">Tank 1</th>
      <th style="padding: 0.4rem 0.9rem; text-align: right;">Tank 2</th>
      <th style="padding: 0.4rem 0.9rem; text-align: right;">Tank 3</th>
      <th style="padding: 0.4rem 0.9rem; text-align: right;">Tank 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 0.3rem 0.9rem;">A<sub>i</sub> &nbsp;(cm²)</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">50.27</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">50.27</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">28.27</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">28.27</td>
    </tr>
    <tr>
      <td style="padding: 0.3rem 0.9rem;">a<sub>i</sub> &nbsp;(cm²)</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">0.233</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">0.242</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">0.127</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">0.127</td>
    </tr>
    <tr>
      <td style="padding: 0.3rem 0.9rem;">x<sub>i</sub><sup>ss</sup> &nbsp;(cm)</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">14.0</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">14.0</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">14.2</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">21.3</td>
    </tr>
    <tr>
      <td style="padding: 0.3rem 0.9rem;">x<sub>i</sub><sup>L</sup>, x<sub>i</sub><sup>U</sup> &nbsp;(cm)</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">7.5, 28.0</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">7.5, 28.0</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">3.5, 28.0</td>
      <td style="padding: 0.3rem 0.9rem; text-align: right;">4.5, 28.0</td>
    </tr>
  </tbody>
</table>

<div style="font-size: 0.95rem; color: #495057;">
γ<sub>1</sub> = γ<sub>2</sub> = 0.4 &nbsp;·&nbsp;
g = 981 cm/s² &nbsp;·&nbsp;
u<sub>1</sub><sup>ss</sup> = 43.4 ml/s, u<sub>2</sub><sup>ss</sup> = 35.4 ml/s &nbsp;·&nbsp;
0 ≤ u<sub>k</sub> ≤ 60 ml/s
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(r"""
### Deviation variables

The optimal control problem regulates the system about its steady state.
Define deviations from the operating point:

$$z_i = x_i - x_i^{ss}, \qquad v_k = u_k - u_k^{ss}$$

so that $z = 0,\; v = 0$ corresponds to the setpoint
$(x^{ss}, u^{ss})$.

### Optimal control problem

Although discretized ODE systems are convenient to implement for digital
control systems, they are not very convenient to write down cleanly.
Consider the discrete time formulation as an approximation of the
following continuous control problem.

Given an initial state $x(0) = x^0$ from the sidebar sliders, find pump
trajectories that drive the system back to steady state:

$$\min_{x(\cdot),\, u(\cdot)} \; \int_0^T \left( \sum_{i=1}^{4} z_i(t)^2 \;+\; \rho \sum_{k=1}^{2} v_k(t)^2 \right) dt$$

subject to

- the four ODEs above,
- the deviation-variable definitions $z_i = x_i - x_i^{ss}$ and $v_k = u_k - u_k^{ss}$,
- the initial condition $x_i(0) = x_i^0$,
- pump bounds $0 \le u_k(t) \le 60$ ml/s,
- and per-tank bounds $x_i^L \le x_i(t) \le x_i^U$ with the values tabulated above.

The horizon $T$, discretization, and control-penalty weight $\rho$ are
set in the sidebar.

### Solution method

The state and control trajectories are discretized using orthogonal
collocation on finite elements (Radau-IIA, 3 collocation points),
following the simultaneous direct-transcription approach in Biegler
(2010, ch. 10). The resulting nonlinear program is solved with POUNCE,
a Rust reimplementation of the IPOPT primal-dual interior-point algorithm.

See the [companion Jupyter notebook](https://github.com/devin-griff/quad-tank/blob/main/Quad%20tank%20open%20loop.ipynb)
for the full Pyomo implementation.

### References

[1] T. Raff, S. Huber, Z. K. Nagy, and F. Allgöwer, "Nonlinear Model
Predictive Control of a Four Tank System: An Experimental Stability
Study," in *Proc. 2006 IEEE Int. Conf. on Control Applications*, Munich,
Germany, 2006, pp. 237–242.
[IEEE Xplore](https://ieeexplore.ieee.org/document/4776652)

[2] L. T. Biegler, *Nonlinear Programming: Concepts, Algorithms, and
Applications to Chemical Processes*. Philadelphia, PA: SIAM, 2010.
[SIAM](https://epubs.siam.org/doi/book/10.1137/1.9780898719383)

[3] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird,
B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff,
*Pyomo: Optimization Modeling in Python*, 3rd ed. Cham: Springer,
2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
""")


# ── Main layout ───────────────────────────────────────────────────────────────
#
# Module-level code runs on every Streamlit rerun. The flow:
#   1. Manual solve. If the user clicks "Run Optimizer", run the solver
#      with the current sidebar values, stash the result, and rerun so
#      the rest of the script renders against it. No first-load
#      auto-solve: the page opens cold and waits for the user to click.
#   2. Toast. After a successful solve we set `solve_status`; on the next
#      rerun (after the `st.rerun()` above) we pop it and show a toast.
#   3. Tabs. Simulation (animated schematic), Plots (time series),
#      Formulation, Logs. Formulation always renders; the others fall
#      back to an info message until the first solve.

# Manual solve in response to the sidebar button.
if solve_btn:
    with st.spinner("Running POUNCE optimization..."):
        try:
            res = solve_model([z1init, z2init, z3init, z4init], nfe, h_step, rho)
        except Exception as e:
            st.error(f"Solver error: {e}")
            st.stop()

    res["inputs"] = current_inputs
    st.session_state["res"] = res
    st.session_state["solve_status"] = res["status"]
    st.session_state["autoplay"] = True
    st.rerun()  # clean re-render: lands on Simulation tab, no spinner blocking charts

# Surface a toast only when the solver returns a non-optimal status: the
# happy path stays silent so solves don't spam the user. `pop` ensures the
# warning fires once per solve (on the rerun immediately after).
_status = st.session_state.pop("solve_status", None)
if _status is not None and _status != "optimal":
    st.toast(f"Solver status: {_status}: results may be inaccurate.", icon="⚠️")

st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Quad Tank Control: Open Loop Dynamic Optimization "
    "<a href='https://github.com/devin-griff/quad-tank' target='_blank' "
    "title='View source on GitHub' "
    "style='display: inline-block; vertical-align: 0.02em; margin: 0 0.35rem 0 0.1rem; "
    "color: inherit;'>"
    "<svg viewBox='0 0 16 16' width='20' height='20' fill='currentColor' "
    "aria-label='GitHub'>"
    "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17."
    "55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
    ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
    "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59."
    "82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
    ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51"
    ".56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1."
    "07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-"
    "8-8-8z'/></svg></a>"
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/jkitchin/pounce' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>POUNCE</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([6, 3])
with _caption_col:
    st.markdown(
        "Simulate open-loop optimal control of the quadruple-tank process: set "
        "the four initial tank heights with the sidebar sliders and click "
        "**Run Optimizer** to compute pump trajectories that drive the "
        "system back to steady state. The **Simulation** tab animates the "
        "result, **Plots** shows the time-series, **Formulation** explains "
        "the model, and **Logs** shows POUNCE's output."
    )

# Four tabs: animated schematic, time series, formulation/references, solver log.
tab_sim, tab_plots, tab_form, tab_logs = st.tabs(
    ["▶  Simulation", "📈  Plots", "📐  Formulation", "📋  Logs"]
)

res = st.session_state.get("res")

with tab_sim:
    if has_pending_changes:
        # Run Optimizer is enabled: the sidebar holds an initial condition
        # that hasn't been solved (cold load, or inputs changed since the
        # last solve). Preview it: a static pseudo-res holding the slider
        # heights across every frame with idle pumps, so the schematic
        # shows the tanks where the sliders put them. Same shape as a real
        # solve, so build_tank_figure needs no special-casing; the
        # Play/Pause/slider controls are muted by the post-render JS below
        # since there's no trajectory to animate.
        n_state = int(nfe) + 1
        n_input = int(nfe)
        preview_res = {
            "h": float(h_step),
            "t": [k * float(h_step) for k in range(n_state)],
            "z10": [z1init] * n_state,
            "z20": [z2init] * n_state,
            "z30": [z3init] * n_state,
            "z40": [z4init] * n_state,
            "v1": [0.0] * n_input,
            "v2": [0.0] * n_input,
        }
        st.plotly_chart(
            build_tank_figure(preview_res), use_container_width=True
        )
    else:
        st.plotly_chart(build_tank_figure(res), use_container_width=True)

with tab_plots:
    if res is not None:
        st.plotly_chart(build_timeseries(res), use_container_width=True)
    else:
        st.info("Click **Run Optimizer** to see time-series plots.")

with tab_form:
    # Formulation is static reference material: always available, no
    # solve required.
    render_formulation_tab()

with tab_logs:
    if res is not None:
        # pounce's stdout was captured into the `log` field by `solve_model`.
        log = res.get("log", "")
        if log.strip():
            st.code(log, language=None)
        else:
            st.info("No log output was captured. The solver may be writing directly to the system stdout.")
    else:
        st.info("Click **Run Optimizer** to see solver logs.")

# Post-render JS doing four things, all keyed to flags set elsewhere on
# this run:
#   1. Autoplay: when `_should_autoplay` is true (set after a solve),
#      simulate a click on Plotly's Play button so the animation starts
#      without the user pressing it.
#   2. Play-button highlight: when `_highlight_play` is true (set when
#      the sliders match the cached `res`, i.e. nothing new to solve),
#      recolor the Play button to Streamlit's primary red so the user
#      sees that replay is the live action paired with the now-disabled
#      Run Optimizer button in the sidebar.
#   3. Mute Play/Pause/slider while the schematic is a static slider-
#      preview (cold load, or sidebar inputs changed since the last
#      solve): toggled via the `qt-controls-disabled` class.
#   4. Grey the Play button while animation is actively playing (the
#      live action then is Pause, not Play) and snap back to frame 0
#      when a full play-through ends: toggled via `qt-play-active`.
#
# Each toggled effect (2, 3, 4) is delivered by a CSS rule with
# `!important` injected into the parent document, then turned on/off by
# class names on the Play button's wrapping <g> or the slider container.
# Going through Plotly's `updatemenu.bgcolor` doesn't work: that
# attribute sets only the rect stroke in this Plotly version, leaving
# the fill at the active-button default: and patching the rect's
# inline style directly gets wiped on every Plotly redraw during
# animation. A CSS rule with `!important` wins over Plotly's inline
# style, and putting the class on <g> (rather than the rect) keeps it
# attached even if Plotly recreates the inner rect each frame.
#
# Rendered unconditionally so Streamlit's component diff treats this as
# one stable element rather than a fresh iframe per solve. Known
# cosmetic issue: the iframe wrapper claims a few pixels of layout
# space, which can shift the chart slightly on each solve: tracked
# separately.
_should_autoplay = st.session_state.pop("autoplay", False)
# Highlight Play only when there's a real solved result (otherwise the
# chart is in slider-preview mode and the controls are muted).
_highlight_play = (res is not None) and (not has_pending_changes)
# Mute Play/Pause/slider whenever the schematic is a static slider
# preview (cold load, or sidebar inputs changed since the last solve) -
# kept visible so the page doesn't feel empty, but not clickable since
# there's no animation to drive.
_disable_controls = has_pending_changes
components.html(f"""
<script>
(function() {{
    const shouldAutoplay = {str(_should_autoplay).lower()};
    const highlightPlay = {str(_highlight_play).lower()};
    const disableControls = {str(_disable_controls).lower()};
    const doc = window.parent.document;

    // Inject the highlight + disable stylesheet once per session.
    // Idempotent: re-renders just no-op past the existence check.
    if (!doc.getElementById('quad-tank-play-highlight-style')) {{
        const styleEl = doc.createElement('style');
        styleEl.id = 'quad-tank-play-highlight-style';
        styleEl.textContent = `
            .updatemenu-button.qt-play-highlight > rect {{
                fill: #FF4B4B !important;
                stroke: #FF4B4B !important;
            }}
            .updatemenu-button.qt-play-highlight > text {{
                fill: #ffffff !important;
            }}
            /* Visually muted + non-interactive Play/Pause for the
               cold-load steady-state preview. Opacity on the wrapping
               <g> dims rect + text together; pointer-events:none stops
               clicks at the SVG level (Plotly's own handler doesn't see
               the event). */
            .updatemenu-button.qt-controls-disabled {{
                opacity: 0.45;
                cursor: default !important;
                pointer-events: none !important;
            }}
            /* Greys the Play button while the animation is actively
               playing (the available action is Pause, not Play). Same
               treatment as qt-controls-disabled but applied only to
               Play. Toggled by JS on play/pause/end-of-play. */
            .updatemenu-button.qt-play-active {{
                opacity: 0.45;
                cursor: default !important;
                pointer-events: none !important;
            }}
            /* Slider: same treatment. Plotly's slider container class
               is slider-container; muting opacity and blocking pointer
               events stops the drag handle and step ticks from firing
               animate calls. */
            .slider-container.qt-controls-disabled {{
                opacity: 0.45;
                pointer-events: none !important;
            }}
        `;
        doc.head.appendChild(styleEl);
    }}

    // Plotly may not have rendered the buttons / slider by the time this
    // iframe loads on a cold start: the chart drawing is async and can
    // run after our script. Poll every 100 ms (max 30 tries = 3 s) until
    // the Play button exists, then apply the highlight + disabled
    // classes and optionally fire autoplay.
    let tries = 0;
    const findByText = (label) => {{
        for (const t of doc.querySelectorAll('text')) {{
            if (t.textContent.trim() === label) return t.parentElement;
        }}
        return null;
    }};
    const tick = () => {{
        const playBtn = findByText('▶  Play');
        if (!playBtn) {{
            if (++tries < 30) setTimeout(tick, 100);
            return;
        }}
        const pauseBtn = findByText('⏸  Pause');
        const slider = doc.querySelector('.slider-container');

        const setClass = (el, cls, on) => {{
            if (!el) return;
            if (on) el.classList.add(cls);
            else el.classList.remove(cls);
        }};

        setClass(playBtn, 'qt-play-highlight', highlightPlay);
        setClass(playBtn, 'qt-controls-disabled', disableControls);
        setClass(pauseBtn, 'qt-controls-disabled', disableControls);
        setClass(slider, 'qt-controls-disabled', disableControls);

        // Changes 2 + 3 share this plumbing. (2) The Play button greys
        // out while the animation is actively playing; (3) when a full
        // play-through ends, the schematic snaps back to frame 0 so Play
        // can replay from the start. The `__qtPlaying` flag on the graph
        // div distinguishes a Play run from a manual scrub or Pause. End-
        // of-play is detected via `plotly_animatingframe` checking when
        // the active slider step hits the last frame: `plotly_animated`
        // doesn't fire reliably for long autoplay queues in this Plotly
        // build, so we use the per-frame event with both as a belt-and-
        // suspenders trigger. Stale state is cleared on every script run
        // because Streamlit may re-render the chart between solves.
        const gd = doc.querySelector('.js-plotly-plot');
        if (gd) {{
            gd.__qtPlaying = false;
            playBtn.classList.remove('qt-play-active');
        }}
        if (gd && gd.on && !gd._qtAnimResetWired) {{
            gd._qtAnimResetWired = true;
            const setPlayActive = (on) => {{
                const pb = [...doc.querySelectorAll('.updatemenu-button')]
                    .find(b => (b.textContent || '').indexOf('Play') !== -1);
                if (!pb) return;
                if (on) pb.classList.add('qt-play-active');
                else pb.classList.remove('qt-play-active');
            }};
            let resetPending = false;
            const doReset = () => {{
                gd.__qtPlaying = false;
                setPlayActive(false);
                const P = window.parent.Plotly;
                if (P && P.animate) {{
                    P.animate(gd, ['0'], {{
                        mode: 'immediate',
                        frame: {{duration: 0}},
                        transition: {{duration: 0}},
                    }});
                }}
            }};
            const triggerReset = () => {{
                if (resetPending) return;
                resetPending = true;
                setTimeout(() => {{
                    resetPending = false;
                    if (gd.__qtPlaying) doReset();
                }}, 300);
            }};
            gd.addEventListener('click', (e) => {{
                const btn = e.target.closest &&
                            e.target.closest('.updatemenu-button');
                if (!btn) return;
                const txt = btn.textContent || '';
                if (txt.indexOf('Play') !== -1) {{
                    gd.__qtPlaying = true;
                    setPlayActive(true);
                }} else if (txt.indexOf('Pause') !== -1) {{
                    gd.__qtPlaying = false;
                    setPlayActive(false);
                }}
            }}, true);
            gd.addEventListener('mousedown', (e) => {{
                if (e.target && e.target.closest &&
                        e.target.closest('.slider-container')) {{
                    gd.__qtPlaying = false;
                    setPlayActive(false);
                }}
            }}, true);
            const checkAtEnd = () => {{
                if (!gd.__qtPlaying) return;
                const s = gd._fullLayout && gd._fullLayout.sliders &&
                          gd._fullLayout.sliders[0];
                if (!s || !s.steps) return;
                if (s.active === s.steps.length - 1) triggerReset();
            }};
            gd.on('plotly_animatingframe', checkAtEnd);
            gd.on('plotly_animated', checkAtEnd);
        }}

        if (shouldAutoplay && !disableControls) {{
            playBtn.dispatchEvent(new MouseEvent('click', {{bubbles: true}}));
        }}
    }};
    tick();
}})();
</script>
""", height=0)
