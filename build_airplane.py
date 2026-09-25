#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye el objeto ``asb.Airplane`` de AeroSandbox del avión regional
turbohélice (2× motores en el ala alta, 72 pax, fuselaje presurizado,
tren triciclo retráctil) a partir de ``sizing_regional.py``, y genera:

  1) renders/airplane_3view.png  — 3 vistas ortogonales (planta, perfil,
     frente) con matplotlib, a escala uniforme entre paneles, con motores,
     tren, ventanillas, puertas y superficies de control visibles.
  2) renders/airplane_3d.html    — visor 3D interactivo (plotly) para
     rotar el modelo con el mouse.

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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import aerosandbox as asb
import aerosandbox.tools.pretty_plots as p

from sizing_regional import Config, dimensionar

RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
PNG_PATH = os.path.join(RENDER_DIR, "airplane_3view.png")
HTML_PATH = os.path.join(RENDER_DIR, "airplane_3d.html")

# ---------------------------------------------------------------- colores
C_FUSELAGE = "#C9CDD1"  # fuselaje
C_STRUCT = "#9AA3AB"  # góndolas, boquillas de tren, carátulas
C_WING = "#4E79A7"  # ala
C_HT = "#59A14F"  # estabilizador horizontal
C_VT = "#E15759"  # derivada vertical
C_GEAR = "#4A4F57"  # tren de aterrizaje
C_PROP = "#33383F"  # palas de hélice
C_GLASS = "#1F3B57"  # ventanillas y cabina de vuelo
C_LINE_CTRL = "#27313B"  # líneas de superficies de control
C_LINE_DOOR = "#111111"  # puertas y salidas
C_LINE_PROP = "#7A828A"  # disco de hélice

# Geometría de línea (z = 0 es el eje del fuselaje; suelo en z = -2.25)
Z_SUELO = -2.25
Z_ALA = 1.38
Z_HT = 0.10
Z_VT_ROOT = -0.10
Z_MOTOR = 0.95  # eje de hélice (sale de sizing: cfg.z_motor)


# ---------------------------------------------------------------------------
# Construcción del asb.Airplane
# ---------------------------------------------------------------------------
def build_airplane(r: dict, cfg: Config) -> asb.Airplane:
    """Arma el asb.Airplane con las dimensiones de sizing_regional.py."""
    d = cfg.d_fuselaje
    half_d = d / 2
    x_tubo = r["x_tubo_fin"]
    l_fus = r["l_fus"]

    # ------------------------------------------------------------- fuselaje
    # Tubo presurizado Ø d + morro (radoma) + cono con upsweep.
    xf = [
        #  x      ancho   alto    z_c
        (0.00, 0.30, 0.32, 0.02),
        (0.80, 1.05, 1.08, 0.02),
        (1.60, 1.95, 2.00, 0.01),
        (cfg.x_tubo_inicio, d, d, 0.00),
        (5.50, d, d, 0.00),
        (9.00, d, d, 0.00),
        (13.00, d, d, 0.00),
        (17.00, d, d, 0.00),
        (x_tubo, d, d, 0.00),
        (x_tubo + 1.7, 2.15, 2.20, 0.18),
        (x_tubo + 3.3, 1.45, 1.50, 0.45),
        (x_tubo + 4.5, 0.85, 0.90, 0.68),
        (l_fus, 0.55, 0.60, 0.76),
    ]
    fuselaje = asb.Fuselage(
        name="Fuselaje",
        color=C_FUSELAGE,
        xsecs=[
            asb.FuselageXSec(xyz_c=[x, 0, z], width=w, height=h)
            for (x, w, h, z) in xf
        ],
    )

    # -------------------------------------- boquillas de tren (sponsons)
    x_main = r["x_ac"] + cfg.off_main_gear
    sp = []
    for side in (1, -1):
        secs = [
            (x_main - 1.45, 1.30, -0.55, 0.35, 0.55),
            (x_main - 0.85, 1.42, -0.62, 0.75, 1.05),
            (x_main + 0.95, 1.42, -0.62, 0.78, 1.08),
            (x_main + 2.00, 1.26, -0.50, 0.42, 0.64),
        ]
        sp.append(
            asb.Fuselage(
                name="Carena de tren" + (" der." if side > 0 else " izq."),
                color=C_STRUCT,
                xsecs=[
                    asb.FuselageXSec(
                        xyz_c=[x, side * yc, zc], width=w, height=h
                    )
                    for (x, yc, zc, w, h) in secs
                ],
            )
        )

    # ------------------------------------------------------------ góndolas
    y_m = cfg.y_motor
    x_le_nac = cfg.x_le_ala + y_m * math.tan(math.radians(cfg.barrido_le))
    x_prop = x_le_nac - 1.0  # plano de la hélice
    gon = []
    for side in (1, -1):
        secs = [
            (x_prop + 0.08, 0.85, 0.88),
            (x_prop + 0.53, 1.14, 1.16),
            (x_prop + 1.28, 1.18, 1.20),
            (x_prop + 2.38, 1.12, 1.14),
            (x_prop + 3.48, 0.88, 0.90),
            (x_prop + 3.98, 0.50, 0.52),
        ]
        gon.append(
            asb.Fuselage(
                name="Góndola" + (" der." if side > 0 else " izq."),
                color=C_STRUCT,
                xsecs=[
                    asb.FuselageXSec(
                        xyz_c=[x, side * y_m, Z_MOTOR], width=w, height=h
                    )
                    for (x, w, h) in secs
                ],
            )
        )
        # Carátula (spinner)
        gon.append(
            asb.Fuselage(
                name="Carátula" + (" der." if side > 0 else " izq."),
                color=C_STRUCT,
                xsecs=[
                    asb.FuselageXSec(
                        xyz_c=[x, side * y_m, Z_MOTOR], width=w, height=h
                    )
                    for (x, w, h) in [
                        (x_prop - 0.54, 0.10, 0.10),
                        (x_prop - 0.27, 0.55, 0.55),
                        (x_prop - 0.02, 0.72, 0.74),
                    ]
                ],
            )
        )

    # ----------------------------------------------------------------- ala
    s_ala, b = r["area_ala"], r["envergadura"]
    c_root, c_tip = r["c_root"], r["c_tip"]
    half_b = b / 2
    tan_le = math.tan(math.radians(cfg.barrido_le))
    tan_dih = math.tan(math.radians(cfg.diedro))
    af = asb.Airfoil("naca2412")

    ala = asb.Wing(
        name="Ala",
        symmetric=True,
        color=C_WING,
        xsecs=[
            asb.WingXSec(
                xyz_le=[cfg.x_le_ala, 0.0, Z_ALA],
                chord=c_root, twist=0.0, airfoil=af,
            ),
            asb.WingXSec(
                xyz_le=[
                    cfg.x_le_ala + half_b * tan_le,
                    half_b,
                    Z_ALA + half_b * tan_dih,
                ],
                chord=c_tip, twist=cfg.twist_punta, airfoil=af,
            ),
        ],
    )

    # ------------------------------------------------- estabilizador horizontal
    c_root_h = r["c_root_ht"]
    c_tip_h = cfg.lambda_ht * c_root_h
    x_le_ht = r["x_le_ht"]
    htail = asb.Wing(
        name="Estabilizador horizontal",
        symmetric=True,
        color=C_HT,
        xsecs=[
            asb.WingXSec(
                xyz_le=[x_le_ht, 0.0, Z_HT],
                chord=c_root_h, airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[x_le_ht + (r["b_ht"] / 2) * math.tan(math.radians(6)),
                        r["b_ht"] / 2, Z_HT],
                chord=c_tip_h, airfoil=asb.Airfoil("naca0012"),
            ),
        ],
    )

    # ---------------------------------------------------- derivada vertical
    c_root_v = r["c_root_vt"]
    c_tip_v = cfg.lambda_vt * c_root_v
    x_le_vt = r["x_le_vt"]
    h_vt = r["b_vt"]
    vtail = asb.Wing(
        name="Derivada vertical",
        symmetric=False,
        color=C_VT,
        xsecs=[
            asb.WingXSec(
                xyz_le=[x_le_vt, 0.0, Z_VT_ROOT],
                chord=c_root_v, airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[x_le_vt + h_vt * math.tan(math.radians(30)),
                        0.0, Z_VT_ROOT + h_vt],
                chord=c_tip_v, airfoil=asb.Airfoil("naca0012"),
            ),
        ],
    )

    return asb.Airplane(
        name="MiAvion-Regional",
        xyz_ref=[r["x_ac"], 0.0, 0.2],
        wings=[ala, htail, vtail],
        fuselages=[fuselaje] + sp + gon,
        s_ref=s_ala,
        c_ref=r["mac"],
        b_ref=b,
    )


# ---------------------------------------------------------------------------
# Mallas auxiliares (tren, hélices, cristales) y líneas de detalle
# ---------------------------------------------------------------------------
def _cylinder(p0, p1, r, n=14, caps=True):
    """Cilindro (con tapas) de radio r entre los puntos p0 y p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    axis = axis / (L or 1.0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array(
        [0.0, 1.0, 0.0]
    )
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
    a = p0 + r * ring
    b = p1 + r * ring
    pts = np.vstack([a, b])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    if caps:
        c0 = len(pts)
        c1 = c0 + 1
        pts = np.vstack([pts, p0, p1])
        for i in range(n):
            j = (i + 1) % n
            faces += [[c0, j, i], [c1, n + i, n + j]]
    return pts, np.asarray(faces, int)


def gear_meshes(r: dict, cfg: Config):
    """Tren triciclo extendido: struts, ejes y ruedas (gemelas)."""
    out = []
    x_nose = r["x_ac"] - cfg.off_nose_gear
    x_main = r["x_ac"] + cfg.off_main_gear
    z_main = -1.50  # eje principal (r=0.75 → z_suelo=-2.25)
    z_nose = -1.75  # eje nariz (r=0.50)

    # Nariz: par de ruedas gemelas + ballesta
    out.append(_cylinder((x_nose, 0, -1.28), (x_nose, 0, z_nose), 0.10))
    out.append(_cylinder((x_nose, -0.06, -1.30),
                         (x_nose - 0.55, 0, -1.05), 0.05, caps=False))
    for yy in (-0.19, 0.19):
        out.append(_cylinder((x_nose, yy - 0.11, z_nose),
                             (x_nose, yy + 0.11, z_nose), 0.50, n=18))

    # Principales: ballesta desde la carena hasta el eje, ruedas gemelas
    for side in (1, -1):
        out.append(_cylinder(
            (x_main - 0.15, side * 1.40, -0.72),
            (x_main, side * 2.14, z_main), 0.13,
        ))
        for yy in (1.98, 2.30):
            out.append(_cylinder(
                (x_main, side * (yy - 0.14), z_main),
                (x_main, side * (yy + 0.14), z_main), 0.75, n=20,
            ))
    return out  # [(pts, faces)]


def prop_meshes(r: dict, cfg: Config):
    """Palas de las hélices de 6 palas (Ø de cfg.diam_helice)."""
    out = []
    y_m, z_m = cfg.y_motor, Z_MOTOR
    x_le_nac = cfg.x_le_ala + y_m * math.tan(math.radians(cfg.barrido_le))
    x_prop = x_le_nac - 1.0
    rad = cfg.diam_helice / 2
    beta = math.radians(25.0)
    n_blades = cfg.n_palas

    for side in (1, -1):
        c = np.array([x_prop, side * y_m, z_m])
        for i in range(n_blades):
            phi = 2 * np.pi * i / n_blades + 0.3
            r_hat = np.array([0.0, math.cos(phi), math.sin(phi)])
            t_hat = np.array([0.0, -math.sin(phi), math.cos(phi)])
            c_hat = math.cos(beta) * np.array([1.0, 0, 0]) + math.sin(beta) * t_hat
            r0, r1 = 0.28, 1.0
            w0, w1 = 0.34, 0.15
            corners = [
                c + r0 * rad * r_hat - (w0 / 2) * c_hat,
                c + r0 * rad * r_hat + (w0 / 2) * c_hat,
                c + r1 * rad * r_hat + (w1 / 2) * c_hat,
                c + r1 * rad * r_hat - (w1 / 2) * c_hat,
            ]
            tris = [
                [0, 1, 2],
                [0, 2, 3],
            ]
            # doble cara
            tris += [[2, 1, 0], [3, 2, 0]]
            out.append((np.array(corners), np.array(tris, int)))
    return out


def glass_meshes(r: dict, cfg: Config):
    """Ventanillas de pasajeros + cristales de cabina de vuelo."""
    half_d = cfg.d_fuselaje / 2 + 0.02
    out = []

    def tube_patch(xc, half_w, th0, th1, nx=2, nt=2, th_deg=True):
        th0, th1 = math.radians(th0), math.radians(th1)
        out_pts, faces = [], []
        for i in range(nx + 1):
            x = xc - half_w + 2 * half_w * i / nx
            for j in range(nt + 1):
                th = th0 + (th1 - th0) * j / nt
                out_pts.append((x, half_d * math.cos(th), half_d * math.sin(th)))
        idx = lambda i, j: i * (nt + 1) + j
        for i in range(nx):
            for j in range(nt):
                a, b_ = idx(i, j), idx(i, j + 1)
                c_, d_ = idx(i + 1, j + 1), idx(i + 1, j)
                faces += [[a, b_, c_], [a, c_, d_]]
        return np.array(out_pts), np.asarray(faces, int)

    for side in (1, -1):
        pts_all, faces_all = [], []
        base = 0

        def add(pts, faces, side=side):
            nonlocal pts_all, faces_all, base
            q = pts.copy()
            q[:, 1] *= side
            pts_all.append(q)
            faces_all.append(faces + base)
            base += len(q)

        # Ventanillas de pasajeros (una por fila; exclusas en la fila de salida)
        for xw in r["ventanillas_x"]:
            add(*tube_patch(xw, 0.16, 34, 50))
        # Cabina de vuelo: parabrisas envolvente + ventanillas laterales
        add(*tube_patch(3.42, 0.64, 6, 46, nx=3, nt=3))  # parabrisas
        add(*tube_patch(4.58, 0.44, 22, 56, nx=2, nt=3))  # lateral
        if pts_all:
            out.append(
                (np.vstack(pts_all), np.vstack(faces_all))
            )
    # devolvemos un solo lote (ambos costados ya incluidos)
    return out


def door_lines(r: dict, cfg: Config):
    """Contornos de 4 puertas principales + 2 salidas sobre el ala."""
    R = cfg.d_fuselaje / 2 + 0.03
    segs = []

    def loop_x(xc, half_w, th0, th1):
        th = np.radians(np.linspace(th0, th1, 14))
        pts = []
        # borde inferior->superior (x=xc-half_w), superior (arriba), etc.
        left = [(xc - half_w, R * math.cos(t), R * math.sin(t)) for t in th]
        right = [(xc + half_w, R * math.cos(t), R * math.sin(t)) for t in th]
        pts = left + right[::-1] + [left[0]]
        return np.array(pts)

    for side in (1, -1):
        for xc in (r["x_puerta_delantera"], r["x_puerta_trasera"]):
            pts = loop_x(xc, 0.40, -30, 46)
            pts[:, 1] *= side
            segs.append(pts)
        # Salida de emergencia sobre el ala (más chica, más arriba)
        pts = loop_x(r["x_salida_ala"], 0.30, 14, 55)
        pts[:, 1] *= side
        segs.append(pts)
    return segs


def control_lines(r: dict, cfg: Config):
    """Líneas de paneles: slats, flaps, ailerones, spoilers, elevador, timón."""
    segs = []

    def naca_t(frac, tc=0.12):
        x = min(max(frac, 0.0), 1.0)
        return 5 * tc * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

    x_le, z_le = cfg.x_le_ala, Z_ALA
    b2 = r["envergadura"] / 2
    tan_le = math.tan(math.radians(cfg.barrido_le))
    tan_dih = math.tan(math.radians(cfg.diedro))

    def wing_pt(side, y, frac):
        xle = x_le + y * tan_le
        c = r["c_root"] + (r["c_tip"] - r["c_root"]) * (y / b2)
        z = z_le + y * tan_dih + naca_t(frac) * c * 0.70
        return (xle + frac * c, side * y, z)

    def span_line(y0, y1, frac, n=16):
        for side in (1, -1):
            ys = np.linspace(y0, y1, n)
            segs.append(np.array([wing_pt(side, y, frac) for y in ys]))

    def rect_span(y0, y1, f0, f1):
        for side in (1, -1):
            segs.append(np.array([wing_pt(side, y0, f) for f in (f0, f1)]))
            segs.append(np.array([wing_pt(side, y1, f) for f in (f0, f1)]))
            segs.append(np.array([wing_pt(side, y, f0)
                                  for y in np.linspace(y0, y1, 8)]))
            segs.append(np.array([wing_pt(side, y, f1)
                                  for y in np.linspace(y0, y1, 8)]))

    span_line(2.0, b2 - 0.55, 0.13)  # riel de slats
    span_line(2.0, 8.6, 0.70)  # bisagra de flaps
    span_line(9.4, b2 - 0.45, 0.70)  # bisagra de ailerones
    rect_span(4.8, 8.6, 0.42, 0.66)  # spoilers

    # Elevador (HT)
    c_root_h, b_ht2 = r["c_root_ht"], r["b_ht"] / 2
    c_tip_h = cfg.lambda_ht * c_root_h
    for side in (1, -1):
        ys = np.linspace(0.2, b_ht2 - 0.1, 14)
        pts = []
        for y in ys:
            xle = r["x_le_ht"] + y * math.tan(math.radians(6))
            c = c_root_h + (c_tip_h - c_root_h) * (y / b_ht2)
            pts.append((xle + 0.60 * c, side * y, Z_HT + 0.06))
        segs.append(np.array(pts))

    # Timón de dirección (VT)
    c_root_v, h_vt = r["c_root_vt"], r["b_vt"]
    c_tip_v = cfg.lambda_vt * c_root_v
    zs = np.linspace(Z_VT_ROOT + 0.30, Z_VT_ROOT + h_vt - 0.12, 12)
    pts = []
    for z in zs:
        xle = r["x_le_vt"] + (z - Z_VT_ROOT) * math.tan(math.radians(30))
        c = c_root_v + (c_tip_v - c_root_v) * ((z - Z_VT_ROOT) / h_vt)
        pts.append((xle + 0.62 * c, 0.0, z))
    segs.append(np.array(pts))
    return segs


def prop_disk_lines(r: dict, cfg: Config):
    """Círculo del disco barrido de cada hélice."""
    x_le_nac = cfg.x_le_ala + cfg.y_motor * math.tan(
        math.radians(cfg.barrido_le)
    )
    x_prop = x_le_nac - 1.0
    rad = cfg.diam_helice / 2
    th = np.linspace(0, 2 * np.pi, 72)
    segs = []
    for side in (1, -1):
        c = np.array([x_prop, side * cfg.y_motor, Z_MOTOR])
        ring = np.column_stack([
            np.full_like(th, c[0]),
            c[1] + rad * np.cos(th),
            c[2] + rad * np.sin(th),
        ])
        segs.append(ring)
    return segs


def component_meshes(ap):
    """Mallas de los componentes asb: [(pts, faces, color, nombre)]."""
    out = []
    for w in ap.wings:
        pts, faces = w.mesh_body(method="tri")
        out.append((np.asarray(pts, float), np.asarray(faces, int),
                    w.color, w.name))
    for f in ap.fuselages:
        pts, faces = f.mesh_body(method="tri")
        out.append((np.asarray(pts, float), np.asarray(faces, int),
                    f.color, f.name))
    return out


def full_meshes(ap, r: dict, cfg: Config):
    """Todos los sólidos: asb + tren + hélices + cristales."""
    out = component_meshes(ap)
    for pts, faces in gear_meshes(r, cfg):
        out.append((pts, faces, C_GEAR, "Tren de aterrizaje"))
    for pts, faces in prop_meshes(r, cfg):
        out.append((pts, faces, C_PROP, "Hélices"))
    for pts, faces in glass_meshes(r, cfg):
        out.append((pts, faces, C_GLASS, "Ventanillas y cabina"))
    return out


def line_groups(r: dict, cfg: Config):
    """Grupos de líneas: (segmentos, color, dash, nombre)."""
    return [
        (control_lines(r, cfg), C_LINE_CTRL, "dash",
         "Superficies de control"),
        (door_lines(r, cfg), C_LINE_DOOR, "solid", "Puertas y salidas"),
        (prop_disk_lines(r, cfg), C_LINE_PROP, "dot", "Disco de hélice"),
    ]


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def _global_bounds(meshes):
    mins = np.min([m[0].min(axis=0) for m in meshes], axis=0)
    maxs = np.max([m[0].max(axis=0) for m in meshes], axis=0)
    mins[2] = min(mins[2], Z_SUELO - 0.1)  # incluir tren
    pad = 0.04 * (maxs - mins) + 0.05
    return mins - pad, maxs + pad


def save_three_view(ap, r: dict, cfg: Config, path: str) -> None:
    """PNG con 3 vistas ortogonales a escala uniforme (planta/perfil/frente)."""
    meshes = full_meshes(ap, r, cfg)
    groups = line_groups(r, cfg)

    all_tris, all_colors = [], []
    for pts, faces, color, _ in meshes:
        all_tris.append(pts[faces])
        all_colors.append(np.tile(to_rgba(color), (len(faces), 1)))
    tris = np.concatenate(all_tris)
    colors = np.concatenate(all_colors)

    mins, maxs = _global_bounds(meshes)
    spans = maxs - mins
    box = spans / spans.max()

    views = [("Planta", "XY", "z"), ("Perfil", "XZ", "y"),
             ("Frente", "-YZ", "x")]
    imgs = []
    for title, angle, degen in views:
        fig = plt.figure(figsize=(5.6, 6.6), dpi=170, facecolor="white")
        ax = fig.add_axes([0.07, 0.05, 0.90, 0.86], projection="3d")
        ax.set_proj_type("ortho")
        ax.set_facecolor("white")  # asb cambia axes.facecolor a lavanda

        ax.add_collection(Poly3DCollection(
            tris, facecolors=colors,
            edgecolors=(0, 0, 0, 0.10), linewidths=0.2,
            alpha=1.0, shade=True, zsort="average",
        ))
        for segs, color, dash, _ in groups:
            style = {"solid": "-", "dash": "--", "dot": ":"}[dash]
            for seg in segs:
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        color=color, linestyle=style, linewidth=1.1,
                        zorder=5)

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect(box)

        plt.sca(ax)
        p.set_preset_3d_view_angle(angle)
        {"x": ax.xaxis, "y": ax.yaxis, "z": ax.zaxis}[degen].set_ticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor("0.85")
        ax.tick_params(labelsize=8, colors="0.35")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)

    fig = plt.figure(figsize=(17.6, 7.6), dpi=170, facecolor="white")
    for i, img in enumerate(imgs):
        ax = fig.add_axes([0.014 + i * 0.328, 0.10, 0.318, 0.82])
        ax.imshow(img, aspect="equal", interpolation="antialiased")
        ax.set_axis_off()

    fig.suptitle(
        f"{ap.name} — avión regional turbohélice de pasajeros · 3 vistas",
        fontsize=16, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.045,
        (
            f"MTOW ≈ {r['mtow']:.0f} kg   ·   S = {r['area_ala']:.1f} m²   ·   "
            f"b = {r['envergadura']:.2f} m   ·   L = {r['l_fus']:.2f} m   ·   "
            f"2 × {r['p_to_motor']:.0f} kW   ·   V_S(flaps) = "
            f"{r['prestaciones']['SABE']['vs_ld_mlw'] * 1.94384:.0f} kt   ·   "
            "escala uniforme entre vistas"
        ),
        ha="center", fontsize=10.5, color="dimgray",
    )
    fig.text(
        0.5, 0.018,
        (
            "Fuselaje presurizado Ø2,87 m (18 filas 2+2) · ala alta con "
            "slats, flaps, ailerones y spoilers · 2 hélices de 6 palas · "
            "tren triciclo retráctil · 4 puertas + 2 salidas sobre el ala"
        ),
        ha="center", fontsize=9.5, color="dimgray",
    )
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def save_interactive_html(ap, r: dict, cfg: Config, path: str) -> None:
    """HTML interactivo (plotly) para rotar el modelo 3D con el mouse."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Sólidos agrupados por nombre/color
    batches: dict = {}
    for pts, faces, color, name in full_meshes(ap, r, cfg):
        batches.setdefault((name, color), []).append((pts, faces))
    for (name, color), items in batches.items():
        pts = np.vstack([p for p, _ in items])
        faces = np.vstack(
            [f + off for (p, f), off in
             zip(items, np.cumsum([0] + [len(p) for p, _ in items[:-1]]))]
        )
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, flatshading=True, name=name, showlegend=True,
            opacity=1.0,
            lighting=dict(ambient=0.55, diffuse=0.65, specular=0.35,
                          roughness=0.45),
            hovertemplate=("x=%{x:.2f} m, y=%{y:.2f} m, z=%{z:.2f} m"
                           "<extra>%{fullData.name}</extra>"),
        ))

    # Líneas (superficies de control, puertas, discos)
    dash_map = {"dash": "dash", "solid": "solid", "dot": "dot"}
    for segs, color, dash, name in line_groups(r, cfg):
        for k, seg in enumerate(segs):
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(color=color, width=4,
                          dash=dash_map[dash]),
                name=name, showlegend=(k == 0),
                hoverinfo="skip",
            ))

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
            xaxis=axis, yaxis=axis, zaxis=axis,
            camera=dict(eye=dict(x=-1.5, y=-1.25, z=0.55),
                        projection=dict(type="orthographic")),
        ),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, b=0, t=45),
        paper_bgcolor="white",
    )
    fig.write_html(path, include_plotlyjs=True,
                   config=dict(displaylogo=False, responsive=True))


def main() -> None:
    cfg = Config()
    r = dimensionar(cfg=cfg)
    ap = build_airplane(r, cfg)

    w = ap.wings[0]
    print("Airplane construido:", ap.name)
    print(f"  Ala   : S = {w.area():.1f} m² (obj. {r['area_ala']:.1f}) | "
          f"b = {w.span():.2f} m (obj. {r['envergadura']:.2f}) | "
          f"MAC = {w.mean_aerodynamic_chord():.2f} m (obj. {r['mac']:.2f})")
    print(f"  Fuselaje: L = {ap.fuselages[0].length():.2f} m "
          f"(obj. {r['l_fus']:.2f}) · Ø {cfg.d_fuselaje:.2f} m")
    print(f"  Motores : 2 × {r['p_to_motor']:.0f} kW, hélices Ø "
          f"{cfg.diam_helice:.2f} m a y = ±{cfg.y_motor:.2f} m")
    print(f"  MTOW {r['mtow']:.0f} kg · V_S(flaps) "
          f"{r['prestaciones']['SABE']['vs_ld_mlw'] * 1.94384:.0f} kt · "
          f"despegue OEI {r['prestaciones']['SABE']['tofl_oei']:.0f} m "
          f"/ aterrizaje {r['prestaciones']['SABE']['ld_total']:.0f} m")

    os.makedirs(RENDER_DIR, exist_ok=True)
    save_three_view(ap, r, cfg, PNG_PATH)
    print(f"\nPNG  3 vistas : {PNG_PATH}")
    save_interactive_html(ap, r, cfg, HTML_PATH)
    print(f"HTML 3D       : {HTML_PATH}")


if __name__ == "__main__":
    main()
