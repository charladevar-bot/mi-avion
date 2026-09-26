#!/usr/bin/env python3
"""Modelo sólido CAD del avión regional con doble curvatura -> STEP + render isométrico.

Geometría de referencia: sizing del commit d9af33a (MTOW 23553 kg, ala corrida
x_LE 12,56 m) y estaciones de build_airplane.py.

Núcleo de modelado: build123d / OCP (OpenCASCADE) — el mismo kernel B-rep que
usa FreeCAD (instalado vía pip porque apt y conda-forge no están accesibles en
este sandbox). Salidas: STEP ISO-10303 legible en cualquier CAD + render PNG.

Prioridad del pedido: fuselaje de doble curvatura > ventanillas y puertas >
interior de cabina > tren de aterrizaje.
"""
from __future__ import annotations

import math
import os

from build123d import (
    Align,
    Axis,
    Box,
    Compound,
    Cylinder,
    Edge,
    Location,
    Pos,
    Rot,
    Shape,
    Solid,
    Vector,
    Wire,
    export_step,
    import_step,
)

# --------------------------------------------------------------------------
# Constantes de la aeronave (sizing d9af33a / build_airplane)
# --------------------------------------------------------------------------
L_AVION = 26.28                    # longitud total [m]
R_FUS = 1.435                      # radio fuselaje (Ø 2,87) [m]
X_TUBE0, X_TUBE1 = 2.60, 21.48     # tramo cilíndrico [m]
TAIL_UP = 0.55                     # upsweep del cono de cola [m]
WALL = 0.09                        # espesor de pared [m]
Z_SUELO = -2.25                    # plano de tierra (tren extendido)
Z_PISO = -0.42                     # piso de cabina
X_LE_ALA = 12.56                   # borde de ataque del ala (corregido por CG)
S_ALA, B_ALA = 61.2, 26.54         # superficie y envergadura
Z_ALA = 1.38                       # plano del ala (montaje alto)
LAMBDA = 0.35                      # relación de aspectos punta/raíz
SWEEP_LE = 15.0                    # barrido del borde de ataque [°]
Z_MOTOR = 0.95                     # eje de hélice
X_HELICE, R_HELICE = 12.20, 0.66   # disco de hélice (bordes 11,54/12,86)
Y_MOTOR = 4.30                     # semidistancia de los motores
X_PUERTA1, X_PUERTA2 = 5.55, 20.83  # puertas principales
PUERTA_HW, PUERTA_Z0, PUERTA_Z1 = 0.40, -0.95, 1.15
X_EXIT = 14.16                     # salidas sobre el ala
EXIT_HW, EXIT_Z0, EXIT_Z1 = 0.30, 0.15, 1.05
Z_VENT, R_VENT = 0.85, 0.16        # eje y radio de ventanillas (Ø 0,32)
X_FILA0, PITCH, FILAS = 6.75, 0.81, 18   # filas 2+2 de la cabina
X_NARIZ_GEAR = 8.29
R_NARIZ, R_MAIN = 0.50, 0.75
X_MAIN, Y_MAIN = 14.54, 2.14

# Paleta idéntica a build_airplane
C_FUS = (0.788, 0.804, 0.820)
C_STRUCT = (0.604, 0.639, 0.671)
C_WING = (0.306, 0.475, 0.655)
C_HT = (0.349, 0.631, 0.310)
C_VT = (0.882, 0.341, 0.349)
C_GEAR = (0.290, 0.310, 0.341)
C_PROP = (0.200, 0.220, 0.247)
C_SEAT = (0.184, 0.231, 0.337)
C_INT = (0.902, 0.890, 0.855)
C_DOOR = (0.702, 0.725, 0.749)

N_SECC = 48  # puntos por sección superelipse


# --------------------------------------------------------------------------
# Helpers geométricos
# --------------------------------------------------------------------------
def superellipse(x: float, a: float, b: float, cz: float, n: float = 2.6) -> Wire:
    """Sección transversal superelíptica en el plano YZ (doble curvatura).

    Se dibuja como ARISTA SPLINE PERIÓDICA interpolando los 48 puntos (no un
    polígono de 48 caras): la superficie del loft queda realmente lisa.
    """
    pts = []
    for i in range(N_SECC):
        t = 2 * math.pi * i / N_SECC
        c, s = math.cos(t), math.sin(t)
        y = a * math.copysign(abs(c) ** (2 / n), c)
        z = cz + b * math.copysign(abs(s) ** (2 / n), s)
        pts.append(Vector(x, y, z))
    try:
        return Wire([Edge.make_spline(pts, periodic=True)])
    except Exception:  # noqa: BLE001 - fallback si OCC rechaza el spline
        return Wire.make_polygon(pts, close=True)


def strut(p0, p1, r: float) -> Solid:
    """Cilindro desde p0 hasta p1 (puntales del tren)."""
    d = Vector(*p1) - Vector(*p0)
    length = d.length
    cyl = Cylinder(r, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    ang_z = math.degrees(math.atan2(d.Y, d.X))
    ang_y = math.degrees(math.atan2(math.hypot(d.X, d.Y), d.Z))
    # Rot se aplica de derecha a izquierda: primero la inclinación (Y), luego el
    # acimut (Z), y por último la traslación al pie del puntal.
    return Pos(p0) * Rot(0, 0, ang_z) * Rot(0, ang_y, 0) * Solid(cyl.wrapped)


def rounded_prism(width: float, height: float, depth: float, radius: float) -> Shape:
    """Prisma con las 4 aristas longitudinales redondeadas (eje = profundidad)."""
    p = Box(width, depth, height)
    return p.fillet(radius, p.edges().filter_by(Axis.Y))


def _naca_yt(xr: float, chord: float) -> float:
    return 5 * 0.12 * (
        0.2969 * math.sqrt(xr)
        - 0.1260 * xr
        - 0.3516 * xr**2
        + 0.2843 * xr**3
        - 0.1015 * xr**4
    ) * chord


def airfoil_wire(x_le: float, chord: float, z: float, t_rel: float,
                 camber: float = 0.02, n_pt: int = 22, y: float = 0.0) -> Wire:
    """Perfil alar cambrado tipo NACA en la estación y (orden determinista)."""

    def yt(xr: float) -> float:
        return 5 * t_rel * (
            0.2969 * math.sqrt(xr)
            - 0.1260 * xr
            - 0.3516 * xr**2
            + 0.2843 * xr**3
            - 0.1015 * xr**4
        ) * chord

    def yc(xr: float) -> float:
        p, m = 0.4, camber
        if xr < p:
            return m / p**2 * (2 * p * xr - xr**2) * chord
        return m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xr - xr**2) * chord

    xs = [(1 - math.cos(math.pi * i / (n_pt - 1))) / 2 for i in range(n_pt)]
    pts = []
    for xr in xs:  # LE -> TE por la superficie superior
        pts.append(Vector(x_le + chord * xr, y, z + yc(xr) + yt(xr)))
    for xr in reversed(xs):  # TE -> LE por la inferior
        pts.append(Vector(x_le + chord * xr, y, z + yc(xr) - yt(xr)))
    return Wire.make_polygon(pts, close=True)


def wing_station(y: float) -> tuple[float, float, float]:
    """(x_le, chord, z) del ala en la estación y."""
    frac = abs(y) / (B_ALA / 2)
    x_le = X_LE_ALA + abs(y) * math.tan(math.radians(SWEEP_LE))
    chord = (2 * S_ALA / ((1 + LAMBDA) * B_ALA)) * (1 - (1 - LAMBDA) * frac)
    z = Z_ALA + 0.02 * abs(y)  # alabeo leve
    return x_le, chord, z


# --------------------------------------------------------------------------
# Fuselaje
# --------------------------------------------------------------------------
def build_fuselage() -> tuple[Solid, Solid]:
    """Sólido lofted de doble curvatura + cavidad interior + piso/cubierta."""
    outer, inner = [], []

    # Nariz ogiva (0 -> 2,60): doble curvatura en ambos planos.
    # La cavidad arranca en t=1/8 (no coplanar con la tapa exterior): evita que
    # el booleano OCC falle y deja la punta del radomo maciza, como en lo real.
    for i in range(9):
        t = i / 8
        x = X_TUBE0 * t
        s = math.sqrt(max(0.0, 1 - (1 - t) ** 2.2))
        a = max(0.16, R_FUS * s)
        outer.append(superellipse(x, a, a, 0.0))
        if i > 0:
            inner.append(superellipse(x, max(0.06, a - WALL), max(0.06, a - WALL), 0.0))

    # Tubo cilíndrico (Ø 2,87 exacto)
    for x in (3.4, 5.0, 7.0, 9.5, 12.0, 14.5, 17.0, 19.5, X_TUBE1):
        outer.append(superellipse(x, R_FUS, R_FUS, 0.0))
        inner.append(superellipse(x, R_FUS - WALL, R_FUS - WALL, 0.0))

    # Cono de cola con upsweep; la cavidad termina en t=7/8 (punta maciza)
    for i in range(1, 9):
        t = i / 8
        x = X_TUBE1 + (L_AVION - X_TUBE1) * t
        a = R_FUS * (1 - t) + 0.18 * t
        b = R_FUS * (1 - t) + 0.22 * t
        cz = TAIL_UP * t
        outer.append(superellipse(x, a, b, cz))
        if i < 8:
            inner.append(superellipse(x, max(0.05, a - WALL), max(0.05, b - WALL), cz))

    shell = Solid.make_loft(outer, ruled=False)
    cavity = Solid.make_loft(inner, ruled=False)
    body = shell - cavity

    # Piso de cabina (plano z = -0,42) con cuña en la nariz
    floor_main = Pos((2.35 + 21.90) / 2, 0, Z_PISO + 0.04) * Box(21.90 - 2.35, 2.60, 0.08)
    nose_lo = Wire.make_polygon(
        [
            Vector(0.55, -0.50, Z_PISO - 0.04),
            Vector(2.35, -1.30, Z_PISO - 0.04),
            Vector(2.35, 1.30, Z_PISO - 0.04),
            Vector(0.55, 0.50, Z_PISO - 0.04),
        ],
        close=True,
    )
    nose_hi = Wire.make_polygon(
        [
            Vector(0.55, -0.50, Z_PISO + 0.04),
            Vector(2.35, -1.30, Z_PISO + 0.04),
            Vector(2.35, 1.30, Z_PISO + 0.04),
            Vector(0.55, 0.50, Z_PISO + 0.04),
        ],
        close=True,
    )
    floor_nose = Solid.make_loft([nose_lo, nose_hi], ruled=True)
    deck = Pos((2.45 + 5.30) / 2, 0, -0.06) * Box(2.85, 2.30, 0.07)
    floor = floor_main + floor_nose + deck
    return body, floor


# --------------------------------------------------------------------------
# Recortes reales en el fuselaje
# --------------------------------------------------------------------------
def cut_openings(body: Solid) -> tuple[Solid, list[Shape]]:
    """Ventanillas, panorámica de cabina, 4 puertas y 2 salidas (con marcos)."""
    frames: list[Shape] = []
    tools: list[Shape] = []

    # --- Ventanillas de pasajeros (cilindros a lo de Y). Se saltan la salida
    #     sobre el ala y la puerta trasera: quedan 16 repartidas por toda la cabina.
    n_vent = 0
    for i in range(FILAS):
        x = X_FILA0 + PITCH * i
        if abs(x - X_EXIT) < 0.50 or abs(x - X_PUERTA2) < 0.63:
            continue
        tools.append(
            Pos(x, 0, Z_VENT)
            * Cylinder(R_VENT, 3.3, rotation=(90, 0, 0))
        )
        n_vent += 1
    print(f"      ventanillas: {n_vent}")

    # --- Cabina de vuelo: 2 ventanas panorámicas por lado (frontal + lateral)
    for sgn in (1, -1):
        front = Pos(3.05, sgn * 1.05, 0.92) * Rot(0, -9, sgn * 16) * Box(0.95, 0.95, 0.70)
        side = Pos(4.35, sgn * 1.35, 0.80) * Rot(0, -3, sgn * 7) * Box(1.15, 0.40, 0.52)
        tools.extend([front, side])

    # --- 4 puertas principales: recorte + marco en relieve
    for x in (X_PUERTA1, X_PUERTA2):
        for sgn in (1, -1):
            zc = (PUERTA_Z0 + PUERTA_Z1) / 2
            tools.append(
                Pos(x, sgn * 1.45, zc)
                * rounded_prism(2 * PUERTA_HW, PUERTA_Z1 - PUERTA_Z0, 0.55, 0.18)
            )
            outer_f = Pos(x, sgn * 1.44, zc) * rounded_prism(
                2 * PUERTA_HW + 0.14, PUERTA_Z1 - PUERTA_Z0 + 0.14, 0.10, 0.24
            )
            inner_f = Pos(x, sgn * 1.44, zc) * rounded_prism(
                2 * PUERTA_HW, PUERTA_Z1 - PUERTA_Z0, 0.30, 0.18
            )
            frames.append(outer_f - inner_f)

    # --- 2 salidas de emergencia sobre el ala: recorte + marco
    for sgn in (1, -1):
        zc = (EXIT_Z0 + EXIT_Z1) / 2
        tools.append(
            Pos(X_EXIT, sgn * 1.45, zc)
            * rounded_prism(2 * EXIT_HW, EXIT_Z1 - EXIT_Z0, 0.55, 0.10)
        )
        outer_f = Pos(X_EXIT, sgn * 1.44, zc) * rounded_prism(
            2 * EXIT_HW + 0.10, EXIT_Z1 - EXIT_Z0 + 0.10, 0.10, 0.15
        )
        inner_f = Pos(X_EXIT, sgn * 1.44, zc) * rounded_prism(
            2 * EXIT_HW, EXIT_Z1 - EXIT_Z0, 0.30, 0.10
        )
        frames.append(outer_f - inner_f)

    for t in tools:
        body = body - t
    return body, frames


# --------------------------------------------------------------------------
# Interior de cabina
# --------------------------------------------------------------------------
def build_seat() -> Solid:
    cushion = Pos(0, 0, 0.05) * Box(0.50, 0.44, 0.10)
    back = Pos(0.245, 0, 0.36) * Box(0.09, 0.44, 0.62)
    head = Pos(0.245, 0, 0.75) * Box(0.09, 0.28, 0.16)
    arm_l = Pos(0.02, 0.245, 0.24) * Box(0.42, 0.05, 0.06)
    arm_r = Pos(0.02, -0.245, 0.24) * Box(0.42, 0.05, 0.06)
    return cushion + back + head + arm_l + arm_r


def build_interior() -> list[tuple[str, Shape]]:
    items: list[tuple[str, Shape]] = []

    # Asientos 2+2 con pasillo central, 18 filas
    seat_proto = build_seat()
    seats = []
    for i in range(FILAS):
        x = X_FILA0 + PITCH * i
        for y in (-1.00, -0.52, 0.52, 1.00):
            seats.append(Location((x, y, Z_PISO + 0.08)) * seat_proto)
    items.append(("Asientos 2+2 (18 filas)", Compound.make_composite(seats)))

    # Galley delantero (babor, detrás de la puerta 1)
    galley_f = Pos(6.205, -0.95, Z_PISO + 0.86) * Box(0.45, 0.70, 1.72)
    counter_f = Pos(6.205, -0.95, Z_PISO + 1.69) * Box(0.50, 0.74, 0.06)
    items.append(("Galley delantero", galley_f + counter_f))

    # Galley trasero (estribor, con paso libre a la puerta trasera)
    galley_r = Pos(21.15, 0.80, Z_PISO + 0.86) * Box(0.50, 0.50, 1.72)
    counter_r = Pos(21.15, 0.80, Z_PISO + 1.69) * Box(0.55, 0.54, 0.06)
    items.append(("Galley trasero", galley_r + counter_r))

    # Baño (babor, popa): mamparos + puerta entreabierta + inodoro + lavabo
    z0, h = Z_PISO, 1.60
    wall_fwd = Pos(20.90, -0.86, z0 + h / 2) * Box(0.04, 0.88, h)
    wall_side = Pos(21.15, -1.30, z0 + h / 2) * Box(0.50, 0.04, h)
    wall_aft = Pos(21.40, -0.86, z0 + h / 2) * Box(0.04, 0.88, h)
    wall_in = Pos(21.15, -0.42, z0 + h / 2) * Box(0.50, 0.04, h)
    toilet = Pos(21.22, -1.05, z0 + 0.21) * Box(0.34, 0.40, 0.42)
    basin = Pos(21.00, -1.10, z0 + 1.05) * Box(0.26, 0.30, 0.20)
    door = Pos(21.08, -0.42, z0 + 0.78) * Rot(0, 0, -35) * Box(0.42, 0.04, 1.50)
    bath = wall_fwd + wall_side + wall_aft + wall_in + toilet + basin + door
    items.append(("Baño", bath))

    # Asientos de piloto sobre la cubierta de vuelo
    pilot_proto = build_seat()
    pilots = []
    for y in (-0.45, 0.45):
        pilots.append(Location((4.35, y, -0.025)) * pilot_proto)
    items.append(("Asientos de piloto", Compound.make_composite(pilots)))
    return items


# --------------------------------------------------------------------------
# Tren de aterrizaje triciclo extendido, carenas y puertas de compartimiento
# --------------------------------------------------------------------------
def build_gear() -> list[tuple[str, Shape]]:
    items: list[tuple[str, Shape]] = []
    wheels, struts, bays = [], [], []

    # --- Nariz (x 8,29 · r 0,50)
    wheels.append(
        Pos(X_NARIZ_GEAR, 0, Z_SUELO + R_NARIZ)
        * Cylinder(R_NARIZ, 0.30, rotation=(90, 0, 0))
    )
    wheels.append(
        Pos(X_NARIZ_GEAR, 0, Z_SUELO + R_NARIZ)
        * Cylinder(0.17, 0.36, rotation=(90, 0, 0))
    )
    struts.append(
        strut((X_NARIZ_GEAR, 0, -1.00), (X_NARIZ_GEAR, 0, Z_SUELO + R_NARIZ + 0.12), 0.055)
    )
    struts.append(strut((X_NARIZ_GEAR - 0.16, 0, -1.35), (X_NARIZ_GEAR - 0.05, 0, -1.72), 0.030))

    # Compartimiento de nariz + puertas abiertas (partidas)
    bays.append(Pos(X_NARIZ_GEAR, 0, -1.25) * Box(1.00, 0.48, 0.60))
    for sgn in (1, -1):
        d = Box(0.46, 0.26, 0.045)
        d.location = Location((X_NARIZ_GEAR, sgn * 0.27, -1.42)) * Rot(sgn * -95, 0, 0)
        bays.append(d)

    # --- Principales (x 14,54 · y ±2,14 · r 0,75 dobles) con carenas
    for sgn in (1, -1):
        # Carena lateral (sponson) fuselageada con fillets + pozo de tren
        spon = Pos(14.815, sgn * 1.36, -0.95) * Box(3.45, 0.36, 0.90)
        spon = spon.fillet(0.16, spon.edges())
        well = Pos(X_MAIN, sgn * 1.32, -1.28) * Box(1.15, 0.52, 0.72)
        spon = spon - well
        items.append((f"Carena {'estribor' if sgn > 0 else 'babor'}", spon))

        # Puertas del pozo, abiertas (2 por lado)
        for x0 in (13.95, 14.72):
            dd = Box(0.70, 0.36, 0.045)
            dd.location = Location((x0, sgn * 1.32, -1.30)) * Rot(sgn * -75, 0, 0)
            bays.append(dd)

        # Puntales (principal + diagonal) y ruedas dobles
        struts.append(strut((X_MAIN, sgn * 1.18, -0.78), (X_MAIN, sgn * 1.95, -1.50), 0.075))
        struts.append(strut((X_MAIN, sgn * 1.42, -1.44), (X_MAIN, sgn * 2.10, -1.52), 0.050))
        for yw in (sgn * 1.95, sgn * 2.33):
            wheels.append(
                Pos(X_MAIN, yw, Z_SUELO + R_MAIN)
                * Cylinder(R_MAIN, 0.32, rotation=(90, 0, 0))
            )
            wheels.append(
                Pos(X_MAIN, yw, Z_SUELO + R_MAIN)
                * Cylinder(0.26, 0.38, rotation=(90, 0, 0))
            )

    items.append(("Ruedas", Compound.make_composite(wheels)))
    items.append(("Puntales", Compound.make_composite(struts)))
    items.append(("Compartimentos y puertas de tren", Compound.make_composite(bays)))
    return items


# --------------------------------------------------------------------------
# Ala, cola y motores
# --------------------------------------------------------------------------
def build_wing() -> Solid:
    """Ala loftada estación a estación (superficie real, no panel VLM)."""
    ys = [-B_ALA / 2, -8.0, -4.3, -1.5, 0.0, 1.5, 4.3, 8.0, B_ALA / 2]
    sections = []
    for y in ys:
        x_le, chord, z = wing_station(y)
        frac = abs(y) / (B_ALA / 2)
        sections.append(
            airfoil_wire(x_le, chord, z, 0.12 - 0.03 * frac, y=y)
        )
    # ruled: la interpolación B-spline de OCC oscila con estas secciones finas;
    # con paneles entre 9 estaciones el ala queda exacta (11,4 m³) y sin torsión.
    return Solid.make_loft(sections, ruled=True)


def build_tail() -> tuple[Solid, Solid]:
    """EH en z = 0,10 bajo el cono y aleta vertical con barrido."""
    # Estabilizador horizontal: y ±4,3, x_le 22,90, barrido 20°
    eh_secs = []
    for y in (-4.3, -2.0, 0.0, 2.0, 4.3):
        frac = abs(y) / 4.3
        x_le = 22.90 + abs(y) * math.tan(math.radians(20))
        chord = 2.10 * (1 - 0.47 * frac)
        z = 0.10 + 0.015 * abs(y)
        eh_secs.append(airfoil_wire(x_le, chord, z, 0.10, camber=0.0, y=y))
    ht = Solid.make_loft(eh_secs, ruled=False)

    # Aleta vertical: raíz embebida en el cono (z 0,75) hasta z 4,90
    vt_secs = []
    for frac, zc in ((0.0, 0.75), (0.35, 2.20), (0.70, 3.65), (1.0, 4.90)):
        x_le = 22.30 + frac * 2.10
        chord = 3.30 * (1 - 0.55 * frac)
        n_pt = 18
        xs = [(1 - math.cos(math.pi * i / (n_pt - 1))) / 2 for i in range(n_pt)]
        pts = [Vector(x_le + chord * xr, _naca_yt(xr, chord), zc) for xr in xs]
        pts += [Vector(x_le + chord * xr, -_naca_yt(xr, chord), zc) for xr in reversed(xs)]
        vt_secs.append(Wire.make_polygon(pts, close=True))
    vt = Solid.make_loft(vt_secs, ruled=False)
    return ht, vt


def build_engines() -> list[tuple[str, Shape]]:
    items: list[tuple[str, Shape]] = []
    for sgn in (1, -1):
        y = sgn * Y_MOTOR

        # Nacela loftada sobre el eje de hélice + pilón al ala
        secs = []
        for x, r in ((12.25, 0.42), (13.10, 0.47), (14.20, 0.42), (14.90, 0.24)):
            pts = [
                Vector(
                    x,
                    y + r * math.cos(2 * math.pi * i / 40),
                    Z_MOTOR + r * math.sin(2 * math.pi * i / 40),
                )
                for i in range(40)
            ]
            secs.append(Wire.make_polygon(pts, close=True))
        nac = Solid.make_loft(secs, ruled=False)
        spinner = Solid.make_loft(
            [
                Wire.make_polygon(
                    [
                        Vector(
                            11.72,
                            y + 0.16 * math.cos(2 * math.pi * i / 32),
                            Z_MOTOR + 0.16 * math.sin(2 * math.pi * i / 32),
                        )
                        for i in range(32)
                    ],
                    close=True,
                ),
                Wire.make_polygon(
                    [
                        Vector(
                            12.25,
                            y + 0.42 * math.cos(2 * math.pi * i / 32),
                            Z_MOTOR + 0.42 * math.sin(2 * math.pi * i / 32),
                        )
                        for i in range(32)
                    ],
                    close=True,
                ),
            ],
            ruled=False,
        )
        pyl = Pos(13.60, y, 1.24) * Box(1.55, 0.18, 0.55)
        nacelle = nac + spinner + pyl

        # Hélice: 4 palas con paso + buje
        blades = []
        for k in range(4):
            b = (
                Pos(X_HELICE, y, Z_MOTOR)
                * Rot(k * 90, 0, 0)
                * Rot(0, 26, 0)
                * Pos(0, (0.14 + R_HELICE) / 2, 0)
                * Box(0.065, R_HELICE - 0.14, 0.24)
            )
            blades.append(b)
        hub = (
            Pos(X_HELICE - 0.05, y, Z_MOTOR)
            * Rot(0, 90, 0)
            * Cylinder(0.14, 0.22)
        )

        side = "derecho" if sgn > 0 else "izquierdo"
        items.append((f"Motor {side}", nacelle))
        items.append((f"Hélice {side}", Compound.make_composite(blades + [hub])))
    return items


# --------------------------------------------------------------------------
# Render isométrico de alta calidad
# --------------------------------------------------------------------------
def render_iso(parts: list[tuple[str, Shape, tuple]], path: str,
               legend_labels: list | None = None, footer: str | None = None) -> None:
    """Render isométrico propio (numpy + PIL): orden por profundidad por
    triángulo + supersampling x2. Mucho más robusto que el painter de mpl3d."""
    import numpy as np
    from PIL import Image, ImageDraw

    # ---- 1. Teselar todas las piezas -> triángulos (mundo) + color base
    tris: list[np.ndarray] = []
    cols: list[tuple] = []
    for _name, shape, color in parts:
        verts, idx = shape.tessellate(0.015, 0.15)
        pts = np.array([[v.X, v.Y, v.Z] for v in verts], dtype=np.float64)
        for a, b_i, c in idx:
            tris.append(pts[[a, b_i, c]])
            cols.append(color)

    # ---- 2. Cámara isométrica (ortográfica)
    azim, elev = math.radians(-58), math.radians(23)
    target = np.array([12.0, 0.0, 0.4])
    cam_dir = np.array(
        [math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim),
         math.sin(elev)]
    )
    fwd = -cam_dir / np.linalg.norm(cam_dir)          # hacia la escena
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    # ---- 3. Proyectar (supersampling x2)
    ss = 2
    W, H = 15 * 170, int(9.5 * 170)
    span_x, span_y = 46.0, 29.0          # unidades del mundo que caben
    scale = min(W * ss / span_x, H * ss / span_y)
    cx, cy = W * ss / 2, H * ss / 2

    light = np.array([0.45, -0.65, 0.85])
    light = light / np.linalg.norm(light)

    img = Image.new("RGB", (W * ss, H * ss), (247, 248, 249))
    draw = ImageDraw.Draw(img)

    # Suelo primero como losa única: nunca debe ocluir al modelo (todo el
    # avión está por encima de z = -2,25) y aislado llena sin huecos.
    ground = np.array([[-9.0, -18.0, Z_SUELO], [35.0, -18.0, Z_SUELO],
                       [35.0, 18.0, Z_SUELO], [-9.0, 18.0, Z_SUELO]]) - target
    gsx = (ground @ right) * scale + cx
    gsy = -(ground @ up) * scale + cy
    draw.polygon(
        [(float(a), float(b)) for a, b in zip(gsx, gsy)],
        fill=(182, 185, 187),
    )

    order = []
    for k, tri in enumerate(tris):
        centroid = tri.mean(axis=0) - target
        depth = float(centroid @ fwd)        # mayor = más lejos
        order.append((depth, k))
    order.sort(reverse=True)                 # de lejos a cerca

    for depth, k in order:
        tri = tris[k]
        v = tri - target
        sx = (v @ right) * scale + cx
        sy = -(v @ up) * scale + cy
        if sx.max() < 0 or sx.min() > W * ss or sy.max() < 0 or sy.min() > H * ss:
            continue
        # normal para iluminación (doble cara)
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        n = np.cross(e1, e2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        lum = 0.46 + 0.54 * abs(float(n @ light) / nn)
        r, g, b = cols[k]
        color = (
            min(255, int(r * lum * 255)),
            min(255, int(g * lum * 255)),
            min(255, int(b * lum * 255)),
        )
        draw.polygon(
            [(float(sx[0]), float(sy[0])), (float(sx[1]), float(sy[1])),
             (float(sx[2]), float(sy[2]))],
            fill=color,
        )

    img = img.resize((W, H), Image.LANCZOS)  # downsampling = antialias

    # ---- 4. Leyenda + pie con matplotlib encima de la imagen
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(15, 9.5), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    if legend_labels is None:
        legend = [
            Patch(facecolor=C_FUS, label="Fuselaje: sólido lofted de doble curvatura"),
            Patch(facecolor=C_DOOR, label="Puertas, salidas y ventanillas (recortes reales)"),
            Patch(facecolor=C_WING, label="Ala: S 61,2 m² · x_LE 12,56 m (reaubicada por CG)"),
            Patch(facecolor=C_HT, label="Estabilizadores horizontal y vertical"),
            Patch(facecolor=C_GEAR, label="Tren triciclo extendido + carenas"),
            Patch(facecolor=C_SEAT, label="Cabina 2+2 · galley delantero/trasero · baño"),
        ]
    else:
        legend = [Patch(facecolor=c, label=l) for l, c in legend_labels]
    ax.legend(handles=legend, loc="upper left", fontsize=8.5, frameon=True)
    if footer is None:
        footer = (
            "Modelo sólido CAD (OpenCASCADE) · export STEP ISO-10303 · MTOW 23 553 kg · "
            "CG 22,9–27,3 %MAC · bodega 4,5 m³ · 16 ventanillas, 4 puertas y 2 salidas recortadas"
        )
    fig.text(
        0.5,
        0.014,
        footer,
        ha="center",
        fontsize=9,
        color=(0.30, 0.33, 0.36),
    )
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "renders")
    os.makedirs(out_dir, exist_ok=True)

    print("1/5 Fuselaje: loft de doble curvatura + cavidad + piso…")
    body, floor = build_fuselage()
    assert body.is_valid, "fuselaje inválido"
    print(f"      volumen casco = {body.volume:.2f} m³")

    print("2/5 Recortes: ventanillas, panorámica, puertas, salidas…")
    body, frames = cut_openings(body)
    assert body.is_valid, "fuselaje inválido tras recortes"
    print(f"      volumen final casco = {body.volume:.2f} m³ · marcos = {len(frames)}")

    print("3/5 Interior, tren, ala, cola y motores…")
    interior = build_interior()
    gear = build_gear()
    wing = build_wing()
    assert wing.is_valid, "ala inválida"
    ht, vt = build_tail()
    assert ht.is_valid and vt.is_valid, "cola inválida"
    engines = build_engines()

    parts: list[tuple[str, Shape, tuple]] = [
        ("Fuselaje", body, C_FUS),
        ("Piso y cubierta", floor, C_STRUCT),
        ("Marcos de puertas y salidas", Compound.make_composite(frames), C_DOOR),
        ("Ala", wing, C_WING),
        ("Estabilizador horizontal", ht, C_HT),
        ("Aleta vertical", vt, C_VT),
    ]
    for name, sh in interior:
        parts.append((name, sh, C_SEAT if "sientos" in name else C_INT))
    for name, sh in gear:
        parts.append((name, sh, C_GEAR))
    for name, sh in engines:
        parts.append((name, sh, C_PROP if "lice" in name else C_STRUCT))

    print("4/5 Export STEP…")
    step_path = os.path.join(out_dir, "airplane_solid.step")
    export_step(Compound.make_composite([p[1] for p in parts]), step_path)
    size = os.path.getsize(step_path)
    back = import_step(step_path)
    n_solids = len(back.solids())
    print(f"      {step_path} ({size:,} bytes · roundtrip {n_solids} sólidos)")

    print("5/5 Render isométrico…")
    png_path = os.path.join(out_dir, "airplane_iso.png")
    render_iso(parts, png_path)
    print(f"      {png_path} ({os.path.getsize(png_path):,} bytes)")

    print("\nResumen del modelo:")
    for name, sh, _c in parts:
        print(f"  - {name}: {len(sh.solids())} solido(s), V = {sh.volume:.2f} m³")


if __name__ == "__main__":
    main()
