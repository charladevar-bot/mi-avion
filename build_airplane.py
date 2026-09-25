#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye un objeto ``asb.Airplane`` de AeroSandbox (fuselaje, ala,
estabilizador horizontal y derivada vertical) a partir de los resultados de
``sizing_2pax.py``, y genera dos visualizaciones:

  1) renders/airplane_3view.png  — 3 vistas ortogonales (planta, perfil,
     frente) con matplotlib, a escala uniforme entre paneles.
  2) renders/airplane_3d.html    — visor 3D interactivo (plotly) donde el
     modelo puede rotarse con el mouse.

Uso:  .venv/bin/python build_airplane.py
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")  # backend sin display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import aerosandbox as asb
import aerosandbox.tools.pretty_plots as p

from sizing_2pax import Config, dimensionar

RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
PNG_PATH = os.path.join(RENDER_DIR, "airplane_3view.png")
HTML_PATH = os.path.join(RENDER_DIR, "airplane_3d.html")

# Colores (válidos en matplotlib y plotly)
COLOR_FUSELAGE = "#C9CDD1"
COLOR_WING = "#4E79A7"
COLOR_HT = "#59A14F"
COLOR_VT = "#E15759"

# Relaciones de taper de los conjuntos
LAMBDA_WING = 0.60
LAMBDA_HT = 0.70
LAMBDA_VT = 0.60

# Colocación longitudinal (m, nariz en x=0, cola en x=L)
X_LE_WING = 1.25  # borde de ataque del ala (sobre el cortafuegos)
Z_WING = 0.66  # ala alta: línea de cuerda sobre la crujía del fuselaje
DIHEDRAL_DEG = 1.5
TWIST_TIP_DEG = -2.0  # lavado (washout) en punta
Z_HT = 0.05  # estabilizador a media altura del fuselaje
SWEEP_VT_DEG = 12.0  # barrido del borde de ataque de la derivada


def _taper_chords(area: float, span: float, lam: float) -> tuple[float, float]:
    """Cuerda raíz y punta de un trapecio de dada área, envergadura y taper."""
    c_root = 2 * area / (span * (1 + lam))
    return c_root, lam * c_root


def _mac(c_root: float, lam: float) -> float:
    """Cuerda media aerodinámica de un ala trapecial."""
    return (2 / 3) * c_root * (1 + lam + lam**2) / (1 + lam)


def build_airplane(r: dict, cfg: Config) -> asb.Airplane:
    """Arma el asb.Airplane con las dimensiones calculadas en sizing_2pax.py."""
    # ---------------------------------------------------------------- ala
    s_wing, b_wing = r["area_ala"], r["envergadura"]
    c_root_w, c_tip_w = _taper_chords(s_wing, b_wing, LAMBDA_WING)
    mac_w = _mac(c_root_w, LAMBDA_WING)
    half_b = b_wing / 2
    dih = math.radians(DIHEDRAL_DEG)
    af_wing = asb.Airfoil("naca2412")

    wing = asb.Wing(
        name="Ala",
        symmetric=True,
        color=COLOR_WING,
        xsecs=[
            asb.WingXSec(
                xyz_le=[X_LE_WING, 0.0, Z_WING],
                chord=c_root_w,
                twist=0.0,
                airfoil=af_wing,
            ),
            asb.WingXSec(
                xyz_le=[
                    X_LE_WING,
                    half_b,
                    Z_WING + half_b * math.tan(dih),
                ],
                chord=c_tip_w,
                twist=TWIST_TIP_DEG,
                airfoil=af_wing,
            ),
        ],
    )

    # Centro aerodinámico del ala (referencia del avión)
    x_ac = X_LE_WING + mac_w / 4

    # ------------------------------------------------- estabilizador horizontal
    s_ht, b_ht = r["s_ht"], r["b_ht"]
    c_root_h, c_tip_h = _taper_chords(s_ht, b_ht, LAMBDA_HT)
    # Brazo del estabilizador según lo usado en sizing_2pax.py
    x_le_ht = x_ac + cfg.arm_ht * r["l_fus"] - c_root_h / 4

    htail = asb.Wing(
        name="Estabilizador horizontal",
        symmetric=True,
        color=COLOR_HT,
        xsecs=[
            asb.WingXSec(
                xyz_le=[x_le_ht, 0.0, Z_HT],
                chord=c_root_h,
                airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[x_le_ht, b_ht / 2, Z_HT],
                chord=c_tip_h,
                airfoil=asb.Airfoil("naca0012"),
            ),
        ],
    )

    # ------------------------------------------------------ derivada vertical
    s_vt, b_vt = r["s_vt"], r["b_vt"]
    c_root_v, c_tip_v = _taper_chords(s_vt, b_vt, LAMBDA_VT)
    l_fus = r["l_fus"]
    x_le_vt = l_fus - c_root_v  # TE a la altura de la punta del fuselaje
    sweep = math.tan(math.radians(SWEEP_VT_DEG)) * b_vt

    vtail = asb.Wing(
        name="Derivada vertical",
        symmetric=False,
        color=COLOR_VT,
        xsecs=[
            asb.WingXSec(
                xyz_le=[x_le_vt, 0.0, 0.0],
                chord=c_root_v,
                airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[x_le_vt + sweep, 0.0, b_vt],
                chord=c_tip_v,
                airfoil=asb.Airfoil("naca0012"),
            ),
        ],
    )

    # -------------------------------------------------------------- fuselaje
    # Semi-ejes (ancho, alto) según sizing: máx. 1.15 m × 1.30 m en cabina.
    w_max, h_max = cfg.ancho_fus, cfg.alto_fus
    fuselage = asb.Fuselage(
        name="Fuselaje",
        color=COLOR_FUSELAGE,
        xsecs=[
            #           x        ancho                  alto
            asb.FuselageXSec(xyz_c=[0.00, 0, 0], width=0.24, height=0.24),
            asb.FuselageXSec(xyz_c=[0.45, 0, 0], width=0.80, height=0.84),
            asb.FuselageXSec(xyz_c=[1.20, 0, 0], width=1.05, height=1.15),
            asb.FuselageXSec(xyz_c=[2.20, 0, 0], width=w_max, height=h_max),
            asb.FuselageXSec(xyz_c=[3.30, 0, 0], width=1.10, height=1.24),
            asb.FuselageXSec(xyz_c=[4.50, 0, 0], width=0.86, height=1.00),
            asb.FuselageXSec(xyz_c=[5.80, 0, 0], width=0.52, height=0.62),
            asb.FuselageXSec(xyz_c=[l_fus, 0, 0], width=0.10, height=0.12),
        ],
    )

    return asb.Airplane(
        name="MiAvion-2Pax",
        xyz_ref=[x_ac, 0.0, 0.0],
        wings=[wing, htail, vtail],
        fuselages=[fuselage],
        s_ref=s_wing,
        c_ref=r["mac"],
        b_ref=b_wing,
    )


# ---------------------------------------------------------------------------
# Mallas por componente (geometría de asb, estilo propio)
# ---------------------------------------------------------------------------
def component_meshes(
    ap: asb.Airplane,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """[(puntos, caras_triangulares, color, nombre)] por componente."""
    out = []
    for w in ap.wings:
        pts, faces = w.mesh_body(method="tri")
        out.append(
            (np.asarray(pts, float), np.asarray(faces, int), w.color, w.name)
        )
    for f in ap.fuselages:
        pts, faces = f.mesh_body(method="tri")
        out.append(
            (np.asarray(pts, float), np.asarray(faces, int), f.color, f.name)
        )
    return out


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
def save_three_view(ap: asb.Airplane, r: dict, path: str) -> None:
    """PNG con 3 vistas ortogonales (planta, perfil, frente) en matplotlib.

    Cada vista se renderiza en su propia figura con proyección ortográfica y
    el mismo ``box_aspect``, de modo que la escala (px/m) es idéntica entre
    paneles, como en un dibujo de aeronave convencional.
    """
    meshes = component_meshes(ap)

    # Límites globales (comunes a las 3 vistas)
    mins = np.min([m[0].min(axis=0) for m in meshes], axis=0)
    maxs = np.max([m[0].max(axis=0) for m in meshes], axis=0)
    pad = 0.06 * (maxs - mins) + 0.02
    mins, maxs = mins - pad, maxs + pad
    spans = maxs - mins
    box = spans / spans.max()  # proporciones para set_box_aspect

    # Trilas combinados (color por cara) en una sola colección: ordenamiento
    # de profundidad global al dibujar.
    all_tris, all_colors = [], []
    for pts, faces, color, _ in meshes:
        all_tris.append(pts[faces])
        all_colors.append(np.tile(to_rgba(color), (len(faces), 1)))
    tris = np.concatenate(all_tris)
    colors = np.concatenate(all_colors)

    views = [
        ("Planta", "XY", "z"),  # (título, preset, eje degenerado a ocultar)
        ("Perfil", "XZ", "y"),
        ("Frente", "-YZ", "x"),
    ]

    figs_imgs = []
    for title, angle, degen in views:
        fig = plt.figure(figsize=(5.5, 6.4), dpi=170, facecolor="white")
        ax = fig.add_axes([0.07, 0.05, 0.90, 0.86], projection="3d")
        ax.set_proj_type("ortho")
        ax.set_facecolor("white")  # asb cambia axes.facecolor a lavanda

        ax.add_collection(
            Poly3DCollection(
                tris,
                facecolors=colors,
                edgecolors=(0, 0, 0, 0.12),
                linewidths=0.25,
                alpha=1.0,
                shade=True,
                zsort="average",
            )
        )
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect(box)

        plt.sca(ax)
        p.set_preset_3d_view_angle(angle)

        # Ejes: ocultar los ticks del eje degenerado de cada vista
        {"x": ax.xaxis, "y": ax.yaxis, "z": ax.zaxis}[degen].set_ticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor("0.85")
        ax.tick_params(labelsize=8, colors="0.35")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        figs_imgs.append(img)

    # Composición final
    fig = plt.figure(figsize=(17.4, 7.3), dpi=170, facecolor="white")
    for i, img in enumerate(figs_imgs):
        ax = fig.add_axes(
            [0.015 + i * 0.328, 0.075, 0.316, 0.845]
        )
        ax.imshow(img, aspect="equal", interpolation="antialiased")
        ax.set_axis_off()

    fig.suptitle(
        f"{ap.name} — avión liviano de 2 pasajeros · 3 vistas",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"MTOW ≈ {r['mtow']:.0f} kg   ·   OEW ≈ {r['oew']:.0f} kg   ·   "
            f"S = {r['area_ala']:.1f} m²   ·   b = {r['envergadura']:.2f} m   ·   "
            f"L = {r['l_fus']:.2f} m   ·   MAC = {r['mac']:.2f} m   ·   "
            "escala uniforme entre vistas"
        ),
        ha="center",
        fontsize=10.5,
        color="dimgray",
    )
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def save_interactive_html(ap: asb.Airplane, r: dict, path: str) -> None:
    """HTML interactivo (plotly) para rotar el modelo 3D con el mouse."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for pts, faces, color, name in component_meshes(ap):
        fig.add_trace(
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                flatshading=True,
                name=name,
                showlegend=True,
                opacity=1.0,
                lighting=dict(
                    ambient=0.55,
                    diffuse=0.65,
                    specular=0.35,
                    roughness=0.45,
                ),
                hovertemplate=(
                    "x=%{x:.2f} m, y=%{y:.2f} m, z=%{z:.2f} m"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    axis = dict(
        backgroundcolor="rgb(250,250,250)",
        gridcolor="#dddddd",
        showbackground=True,
        zerolinecolor="#cccccc",
        title=dict(text="[m]"),
    )
    fig.update_layout(
        title=(
            f"{ap.name} — 3D interactivo: rotá con el mouse · "
            f"MTOW ≈ {r['mtow']:.0f} kg · b = {r['envergadura']:.2f} m · "
            f"L = {r['l_fus']:.2f} m"
        ),
        scene=dict(
            aspectmode="data",
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            camera=dict(
                eye=dict(x=-1.45, y=-1.15, z=0.55),  # frontal-izquierda-arriba
                projection=dict(type="orthographic"),
            ),
        ),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        paper_bgcolor="white",
    )
    fig.write_html(
        path,
        include_plotlyjs=True,  # autocontenido, sin CDN
        config=dict(displaylogo=False, responsive=True),
    )


def main() -> None:
    cfg = Config()
    r = dimensionar(mtow=600.0, cfg=cfg)
    ap = build_airplane(r, cfg)

    # Verificación: geometría de asb vs. objetivos de sizing
    w = ap.wings[0]
    print("Airplane construido:", ap.name)
    print(f"  Ala      : S = {w.area():.2f} m² (obj. {r['area_ala']:.2f}) | "
          f"b = {w.span():.2f} m (obj. {r['envergadura']:.2f}) | "
          f"MAC = {w.mean_aerodynamic_chord():.2f} m (obj. {r['mac']:.2f})")
    print(f"  Fuselaje : L = {ap.fuselages[0].length():.2f} m "
          f"(obj. {r['l_fus']:.2f})")
    print(f"  Referencia: s_ref = {ap.s_ref:.2f}, c_ref = {ap.c_ref:.2f}, "
          f"b_ref = {ap.b_ref:.2f}")

    os.makedirs(RENDER_DIR, exist_ok=True)
    save_three_view(ap, r, PNG_PATH)
    print(f"\nPNG  3 vistas : {PNG_PATH}")
    save_interactive_html(ap, r, HTML_PATH)
    print(f"HTML 3D       : {HTML_PATH}")


if __name__ == "__main__":
    main()
