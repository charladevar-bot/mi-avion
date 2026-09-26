#!/usr/bin/env python3
"""
Dimensionamiento clase I + estimación de prestaciones de un avión de
pasajeros regional turbohélice (30–90 pasajeros, una sola clase).

Ruta de diseño: Buenos Aires SABE (pista 13/31, 2700 m, elev. 5 m)
            ↔  Rosario      SAAR (pista 02/20, 3000 m, elev. 26 m)

Punto de diseño: 72 pasajeros en configuración 2+2 (18 filas) — dentro del
rango requerido 30–90; versiones acortadas/estiradas del mismo fuselaje
cubren el resto de la familia.

Método
------
* Carga útil: pasajeros + equipaje (95 kg/pax incluido equipaje).
* Combustible: Breguet para hélice en crucero (FL200, 480 km/h TAS) +
  taxi/ascenso + reserva (45 min de espera + 150 km de desvío).
* OEW: suma de componentes con densidades estadísticas de la clase
  (ala con slats/flaps, fuselaje presurizado, colas, tren retráctil,
  2 turbohélices con hélices de 6 palas, sistemas, interior, tripulación).
* Iteración de punto fijo MTOW ↔ estructura ↔ combustible.
* Motores: potencia necesaria derivada del crucero y redondeada a un
  rating estándar; empuje estático por coeficiente de hélice.
* Prestaciones: velocidad de pérdida con flaps (slats + flaps completos),
  distancia de despegue (potencia total y caso motor inoperante) y de
  aterrizaje, comparadas contra las pistas de SABE y SAAR.

Pesos, balance y bodega (esta versión)
--------------------------------------
* Desglose de pesos por componente con SUAVE 2.5.2 (suavecode/SUAVE
  instalado desde GitHub): correlaciones General_Aviation (Raymer) para
  fuselaje, alas, colas, tren y sistemas (con desglose hidráulica /
  eléctrica / aviónica / amueblado / sistema de combustible). El total
  se escala con un único factor para reconciliar con el OEW de diseño
  conservador (la correlación GA subestima fuselajes presurizados de
  >20 m); factor y total sin escala se muestran en el reporte. Si SUAVE
  no está disponible se usa el desglose manual de respaldo.
* Centro de gravedad en tres casos de carga (vacío, 90 pax sin
  equipaje, y 90 pax + equipaje máx. + combustible lleno), verificado
  contra el rango 15–35 % de la MAC; si algún caso queda fuera, la
  posición del ala se itera hasta que los tres entren (motores, tren y
  salidas sobre el ala se desplazan con el ala).
* Bodega de equipaje bajo el piso: 4,5 m³ (90 × 0,05 m³), ubicada lo
  más cerca posible del CG con la menor interferencia con la caja del
  ala; se marca en las vistas.

Uso:  .venv/bin/python sizing_regional.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from suave_pesos import desglose_pesos

G = 9.80665  # m/s²
RHO_SL = 1.225  # kg/m³ (ISA nivel del mar)


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ Ruta
    distancia_ruta_km: float = 330.0
    pista_sabe_m: float = 2700.0
    elev_sabe_m: float = 5.0
    pista_saar_m: float = 3000.0
    elev_saar_m: float = 26.0

    # --------------------------------------------------------------- Cargas
    n_pax: int = 72  # punto de diseño (familia 30–90)
    masa_pax: float = 95.0  # kg/pax con equipaje
    n_pilotos: int = 2
    n_cabina: int = 3  # tripulantes de cabina
    masa_piloto: float = 95.0
    masa_cabina: float = 80.0

    # ---------------------------------------------------------------- Misión
    fl_cruz_m: float = 6096.0  # FL200
    v_cruz_kmh: float = 480.0  # TAS de crucero
    div_km: float = 150.0  # desvío del reserva
    espera_h: float = 0.75  # 45 min de espera

    # ------------------------------------------------ Aerodinámica / consumo
    l_d_cruz: float = 14.5
    eta_prop: float = 0.83
    bsfc: float = 0.27  # kg/(kW·h) eje, turbohélice moderno
    oswald: float = 0.80
    cd0_limpio: float = 0.024
    dcd_tren: float = 0.014
    dcd_flap_to: float = 0.010  # slats + flaps 15°
    dcd_flap_ld: float = 0.058  # slats + flaps 35°
    clmax_limpio: float = 1.45
    clmax_to: float = 2.10  # slats + flaps 15°
    clmax_ld: float = 2.75  # slats + flaps 35° (apoyos y frenado)

    # ------------------------------------------------------------------ Ala
    cargalar: float = 385.0  # kg/m² al MTOW
    aspecto: float = 11.5
    lambda_wing: float = 0.25
    barrido_le: float = 5.0  # °
    diedro: float = 2.0  # °
    twist_punta: float = -3.0  # °
    x_le_ala: float = 8.0  # m (nariz en 0)

    # ---------------------------------------------------------------- Colas
    v_h: float = 1.05
    v_v: float = 0.065
    ar_ht: float = 5.0
    ar_vt: float = 1.4
    lambda_ht: float = 0.40
    lambda_vt: float = 0.50

    # ------------------------------------------------------------ Estructura
    k_ala: float = 65.0  # kg/m² (estructura + slats/flaps/alerones/spoilers)
    k_fuselaje: float = 13.0  # kg/m² de área mojada (tubo presurizado)
    k_empennaje: float = 30.0  # kg/m² de superficie de cola
    f_tren: float = 0.050  # × MTOW (triciclo retráctil)
    m_sistemas: float = 1650.0  # aviónica, hidráulica, eléctrica, hielo, press.
    kg_interior_pax: float = 18.0  # asientos, guarnecido, PSU por pax
    m_interior_fijo: float = 400.0  # cocinas, lavabos, barra
    m_misc: float = 300.0  # fluidos, pintura, herramientas

    # ------------------------------------------------------------ Propulsión
    n_motores: int = 2
    frac_cruz_to: float = 0.65  # potencia de crucero / potencia de despegue
    k_empuje: float = 0.0245  # N estáticos por W de potencia de despegue
    m_motor: float = 550.0  # kg c/u (clase PW127)
    m_helice: float = 300.0  # kg c/u, 6 palas
    m_nacelle: float = 260.0  # kg c/u, góndola y montaje
    diam_helice: float = 3.95  # m
    n_palas: int = 6

    # ----------------------------------------------------------- Tren/rodaje
    mu_to: float = 0.02
    mu_frenado: float = 0.40
    f_reversa: float = 0.06

    # ---------------------------------------------------------------- Cabina
    pitch: float = 0.81  # m entre filas (32 in)
    d_fuselaje: float = 2.87  # m diámetro exterior (2+2)

    # Layout longitudinal (nariz en x = 0)
    x_tubo_inicio: float = 2.60
    x_cabina_inicio: float = 6.15
    x_puerta_delantera: float = 5.55
    x_salida_off: float = 1.60  # salida sobre el ala, desde el borde de ataque
    long_cono: float = 4.80
    x_le_ht_off: float = 0.15  # desde fin de tubo
    x_le_vt_off: float = 1.20  # desde fin de tubo

    # ------------------------------------------------------------ Balance/CG
    pct_mac_min: float = 15.0  # % MAC límite delantero aceptable
    pct_mac_max: float = 35.0  # % MAC límite trasero aceptable
    n_pax_balance: int = 90  # pax para los casos de carga (rango 30–90)
    kg_pax_solo: float = 80.0  # kg/pax sin equipaje (95 − 15)
    kg_equipaje_pax: float = 15.0  # kg de equipaje por pax
    vol_equipaje_pax: float = 0.05  # m³ de equipaje por pax
    z_piso_cabina: float = -0.42  # piso de cabina sobre eje (m)
    fr_util_bodega: float = 0.62  # fracción útil del segmento bajo el piso

    # Estaciones de motores y tren (en función de x_ac)
    y_motor: float = 4.75  # semisustentación de la góndola
    z_motor: float = 0.95  # eje de hélice sobre línea de centro (z=0)
    off_nose_gear: float = 5.50  # x_nariz = x_ac − esto
    off_main_gear: float = 0.75  # x_main  = x_ac + esto


# ---------------------------------------------------------------------------
# Cargas y combustible
# ---------------------------------------------------------------------------
def carga_util(cfg: Config) -> float:
    return cfg.n_pax * cfg.masa_pax


def tripulacion(cfg: Config) -> float:
    return cfg.n_pilotos * cfg.masa_piloto + cfg.n_cabina * cfg.masa_cabina


def combustible(mtow: float, cfg: Config) -> dict:
    """Combustible total (kg): taxi+ascenso, crucero, espera y desvío."""
    w_taxi = 0.022 * mtow  # taxi + ascenso a FL200
    w1 = mtow - w_taxi

    v = cfg.v_cruz_kmh / 3.6
    c_prime = cfg.bsfc / 3.6e6  # kg/J
    k = (cfg.eta_prop * cfg.l_d_cruz) / (c_prime * G)  # m
    w2 = w1 * math.exp(-cfg.distancia_ruta_km * 1e3 / k)
    w_cru = w1 - w2

    # Potencia y caudal de crucero (peso medio)
    w_med = 0.5 * (w1 + w2)
    p_total = w_med * G * v / (cfg.eta_prop * cfg.l_d_cruz)  # W
    ff_cru = cfg.bsfc * p_total / 1000.0  # kg/h

    w_espera = 0.55 * ff_cru * cfg.espera_h  # espera a ~55 % de crucero
    w_div = ff_cru * (cfg.div_km / cfg.v_cruz_kmh)

    return {
        "taxi": w_taxi,
        "crucero": w_cru,
        "espera": w_espera,
        "desvio": w_div,
        "total": w_taxi + w_cru + w_espera + w_div,
        "p_crucero_total": p_total,
        "ff_crucero": ff_cru,
    }


# ---------------------------------------------------------------------------
# Pesos por componente (SUAVE), centro de gravedad y bodega de equipaje
# ---------------------------------------------------------------------------
# Orden de grupos para el reporte: (grupo, nombres de ítem)
GRUPOS_ITEMS = (
    ("Estructura", ("Fuselaje presurizado", "Ala", "Estabilizador horizontal",
                    "Derivada vertical")),
    ("Tren de aterrizaje", ("Tren principal", "Tren de nariz")),
    ("Propulsión", ("Motores, hélices y góndolas",)),
    ("Sistemas", ("Control de vuelo", "Hidráulica", "Aviónica", "Eléctrico",
                  "Climatización", "Amueblado", "Sistema de combustible")),
    ("Operativos", ("Tripulación", "Varios")),
)

_SUAVE_A_ITEM = {
    "estructura_fuselaje": "Fuselaje presurizado",
    "estructura_ala": "Ala",
    "estructura_ht": "Estabilizador horizontal",
    "estructura_vt": "Derivada vertical",
    "tren_principal": "Tren principal",
    "tren_nariz": "Tren de nariz",
    "propulsion": "Motores, hélices y góndolas",
    "sis_control": "Control de vuelo",
    "sis_hidraulica": "Hidráulica",
    "sis_avionica": "Aviónica",
    "sis_electrico": "Eléctrico",
    "sis_clima": "Climatización",
    "sis_amueblado": "Amueblado",
    "sis_combustible": "Sistema de combustible",
}


def _geo_suave(cfg: Config, r: dict) -> dict:
    """Geometría/pesos en convergencia, en el formato que espera SUAVE."""
    mojada = math.pi * cfg.d_fuselaje * r["l_fus"] * 0.92
    return dict(
        mtow=r["mtow"], comb_total=r["comb_total"], s_ala=r["area_ala"],
        b=r["envergadura"], mac=r["mac"], c_root=r["c_root"],
        lambda_wing=cfg.lambda_wing, barrido_le=cfg.barrido_le,
        x_le_ala=r.get("x_le_ala", cfg.x_le_ala), x_ac=r["x_ac"],
        x_le_ht=r["x_le_ht"], x_le_vt=r["x_le_vt"],
        s_ht=r["s_ht"], b_ht=r["b_ht"], c_root_ht=r["c_root_ht"],
        lambda_ht=cfg.lambda_ht,
        s_vt=r["s_vt"], b_vt=r["b_vt"], c_root_vt=r["c_root_vt"],
        lambda_vt=cfg.lambda_vt,
        l_fus=r["l_fus"], d_fuselaje=cfg.d_fuselaje, mojada=mojada,
        masa_grupo=r["masa_grupo"], n_motores=cfg.n_motores,
        n_pax=cfg.n_pax_balance, n_asientos=cfg.n_pax,
    )


def desglose_oew(cfg: Config, r: dict) -> tuple[dict, dict]:
    """
    Desglose de pesos OEW por componente.

    Intenta SUAVE (correlaciones General_Aviation / Raymer, con su
    desglose de sistemas); su total se escala con un único factor para
    reconciliar con el OEW de diseño (conservador, anclado a la familia
    ATR72). Sin SUAVE, usa el desglose manual de respaldo.
    Devuelve ({item: kg}, meta).
    """
    budget = r["oew"] - tripulacion(cfg) - cfg.m_misc
    su = desglose_pesos(_geo_suave(cfg, r))
    if su is not None:
        su_items = {v: su[k] for k, v in _SUAVE_A_ITEM.items()}
        bruto = sum(su_items.values())
        # Solo la estructura (regresión de flota GA) se escala para
        # cerrar contra el OEW de diseño: tren, propulsión y sistemas
        # dependen directamente del MTOW y de inputs, van en valor SUAVE.
        estructura = ("Fuselaje presurizado", "Ala",
                      "Estabilizador horizontal", "Derivada vertical")
        estr_raw = sum(su_items[n] for n in estructura)
        resto_raw = bruto - estr_raw
        k = max((budget - resto_raw) / estr_raw, 0.1)
        masas = {
            n: (m * k if n in estructura else m) for n, m in su_items.items()
        }
        meta = {
            "fuente": "SUAVE 2.5.2 (correlaciones General_Aviation / Raymer)",
            "k_escala": k,
            "suave_raw": bruto,
            "suave_disponible": True,
            "nota_k": "aplicado solo a la estructura",
        }
    else:
        mojada = math.pi * cfg.d_fuselaje * r["l_fus"] * 0.92
        masas = {
            "Fuselaje presurizado": cfg.k_fuselaje * mojada,
            "Ala": cfg.k_ala * r["area_ala"],
            "Estabilizador horizontal": cfg.k_empennaje * r["s_ht"],
            "Derivada vertical": cfg.k_empennaje * r["s_vt"],
            "Tren principal": 0.78 * cfg.f_tren * r["mtow"],
            "Tren de nariz": 0.22 * cfg.f_tren * r["mtow"],
            "Motores, hélices y góndolas": r["masa_grupo"],
            "Control de vuelo": 0.28 * cfg.m_sistemas,
            "Hidráulica": 0.10 * cfg.m_sistemas,
            "Aviónica": 0.24 * cfg.m_sistemas,
            "Eléctrico": 0.18 * cfg.m_sistemas,
            "Climatización": 0.13 * cfg.m_sistemas,
            "Sistema de combustible": 0.07 * cfg.m_sistemas,
            "Amueblado": (cfg.kg_interior_pax * cfg.n_pax
                          + cfg.m_interior_fijo),
        }
        meta = {
            "fuente": "manual (respaldo: SUAVE no disponible)",
            "k_escala": 1.0,
            "suave_raw": None,
            "suave_disponible": False,
        }
    masas["Tripulación"] = tripulacion(cfg)
    masas["Varios"] = cfg.m_misc
    return masas, meta


def _ctx(cfg: Config, r: dict, masas: dict) -> dict:
    """Constantes congeladas para evaluar el balance variando solo x_LE."""
    return dict(
        b=r["envergadura"], mac=r["mac"], c_root=r["c_root"],
        c_root_ht=r["c_root_ht"], c_root_vt=r["c_root_vt"],
        x_le_ht=r["x_le_ht"], x_le_vt=r["x_le_vt"],
        x_tubo_inicio=cfg.x_tubo_inicio, x_tubo_fin=r["x_tubo_fin"],
        rows=r["rows"], comb_total=r["comb_total"], masas=masas,
        mtow=r["mtow"],
    )


def _x_ac(cfg: Config, x_le: float, b: float, mac: float) -> float:
    return (x_le
            + math.tan(math.radians(cfg.barrido_le)) * (b / 2) / 2
            + mac / 4)


def _estaciones(cfg: Config, ctx: dict, x_le: float) -> tuple[float, dict]:
    """
    Estaciones longitudinales de cada componente para una posición dada
    del ala. Motores, tren (por desplazamiento de x_ac) y salidas sobre
    el ala se desplazan con el ala.
    """
    b, mac = ctx["b"], ctx["mac"]
    x_ac = _x_ac(cfg, x_le, b, mac)
    tan_le = math.tan(math.radians(cfg.barrido_le))
    x_prop = x_le + cfg.y_motor * tan_le - 1.0
    x_grp = x_prop + 1.61  # cg del grupo (motor 2,0 / hélice 0,6 / góndola 1,8)
    x_nose = x_ac - cfg.off_nose_gear
    x_main = x_ac + cfg.off_main_gear
    x_ti, x_tf = ctx["x_tubo_inicio"], ctx["x_tubo_fin"]
    l_t, l_c = x_tf - x_ti, cfg.long_cono
    x_fus = (l_t * (x_ti + x_tf) / 2 + 0.4 * l_c * (x_tf + l_c / 2)) / (
        l_t + 0.4 * l_c)
    x_ala = x_le + 0.6 * tan_le * (b / 2) + 0.42 * mac
    x_ht = ctx["x_le_ht"] + 0.45 * ctx["c_root_ht"]
    x_vt = ctx["x_le_vt"] + 0.40 * ctx["c_root_vt"]
    x_av = x_ti + 2.20  # bahía de aviónica tras la cabina de pilotos
    cabin_mid = (cfg.x_cabina_inicio
                 + (ctx["rows"] - 1) / 2 * cfg.pitch + 0.45)
    tr = tripulacion(cfg)
    x_crew = (cfg.n_pilotos * cfg.masa_piloto * 4.0
              + cfg.n_cabina * cfg.masa_cabina * 13.0) / tr
    est = {
        "Fuselaje presurizado": x_fus,
        "Ala": x_ala,
        "Estabilizador horizontal": x_ht,
        "Derivada vertical": x_vt,
        "Tren principal": x_main,
        "Tren de nariz": x_nose,
        "Motores, hélices y góndolas": x_grp,
        "Control de vuelo": x_ac + 1.0,
        "Hidráulica": x_ac,
        "Aviónica": x_av,
        "Eléctrico": x_ac + 1.5,
        "Climatización": x_ac,
        "Amueblado": cabin_mid,
        "Sistema de combustible": x_ac,
        "Tripulación": x_crew,
        "Varios": x_ac,
        "_x_ac": x_ac, "_cabin_mid": cabin_mid, "_x_nose": x_nose,
        "_x_main": x_main,
    }
    return x_ac, est


def _proyectar(c: float, dom: list, huecos: list) -> float:
    """Proyecta c sobre dom evitando los intervalos huecos."""
    a, b = dom

    def inter(h):
        return max(h[0], a), min(h[1], b)

    validos = [h for h in huecos if inter(h)[0] <= inter(h)[1]]

    def ok(p):
        return not any(lo < p < hi for lo, hi in validos)

    cands = [min(max(c, a), b)]
    for lo, hi in validos:
        cands += [lo, hi]
    cands = [p for p in cands if ok(p)]
    return min(cands, key=lambda p: abs(p - c))


def _bodega(cfg: Config, ctx: dict, x_le: float, est: dict,
            x_cg3: float) -> dict:
    """
    Dimensiona la bodega bajo el piso y la ubica lo más cerca posible de
    x_cg3 (evita solo el pozo del tren de nariz; ala alta, la caja de
    ala está en el techo del fuselaje, no en el piso, y el tren
    principal es exterior en carenas).
    """
    R = cfg.d_fuselaje / 2
    zf = cfg.z_piso_cabina
    A = R * R * math.acos(zf / R) - zf * math.sqrt(R * R - zf * zf)
    A_us = A * cfg.fr_util_bodega
    V = cfg.n_pax_balance * cfg.vol_equipaje_pax
    L = max(V / A_us, 0.8)
    L = min(L, 0.55 * (ctx["x_tubo_fin"] - cfg.x_cabina_inicio))
    x_nose = est["_x_nose"]
    dom = [cfg.x_cabina_inicio - 1.5 + L / 2, ctx["x_tubo_fin"] - 0.6 - L / 2]
    huecos = [
        [x_nose - 1.0 - L / 2, x_nose + 1.0 + L / 2],
    ]
    centro = _proyectar(x_cg3, dom, huecos)
    return {
        "V": V, "L": L, "A_us": A_us, "A_seg": A,
        "x0": centro - L / 2, "x1": centro + L / 2, "centro": centro,
        "z0": zf, "z1": -(R - 0.12),
        "masa_max": cfg.n_pax_balance * cfg.kg_equipaje_pax,
    }


def _evaluar_casos(cfg: Config, ctx: dict, est: dict,
                   x_bodega: float) -> tuple[list, float]:
    masas = ctx["masas"]
    x_ac, cabin_mid = est["_x_ac"], est["_cabin_mid"]
    x_mac_le = x_ac - ctx["mac"] / 4
    base = [(masas[n], est[n]) for n in masas]
    n = cfg.n_pax_balance
    defs = [
        ("1 · OEW (vacío, sin comb. ni pax)", []),
        (f"2 · {n} pax sin equipaje",
         [(n * cfg.kg_pax_solo, cabin_mid)]),
        (f"3 · {n} pax + equipaje máx. + comb. lleno",
         [(n * cfg.kg_pax_solo, cabin_mid),
          (n * cfg.kg_equipaje_pax, x_bodega),
          (ctx["comb_total"], x_ac)]),
    ]
    out = []
    for nombre, extra in defs:
        items = base + extra
        w = sum(m for m, _ in items)
        x = sum(m * xx for m, xx in items) / w
        pct = 100 * (x - x_mac_le) / ctx["mac"]
        viol = max(0.0, cfg.pct_mac_min - pct, pct - cfg.pct_mac_max)
        out.append({"nombre": nombre, "peso": w, "x_cg": x, "pct": pct,
                    "viol": viol})
    return out, x_mac_le


def evaluar_balance(cfg: Config, ctx: dict, x_le: float) -> dict:
    """CG de los 3 casos + bodega, con iteración bodega ↔ CG."""
    x_cg3, x_bod, res, bod, est = None, None, None, None, None
    for _ in range(8):
        _, est = _estaciones(cfg, ctx, x_le)
        semilla = x_cg3 if x_cg3 is not None else est["_x_ac"]
        bod = _bodega(cfg, ctx, x_le, est, semilla)
        casos, x_mac_le = _evaluar_casos(cfg, ctx, est, bod["centro"])
        x_cg3 = casos[2]["x_cg"]
        res = {"casos": casos, "x_mac_le": x_mac_le, "bodega": bod,
               "estaciones": est, "x_ac": est["_x_ac"]}
        if x_bod is not None and abs(bod["centro"] - x_bod) < 1e-4:
            break
        x_bod = bod["centro"]
    return res


def _eval_x(cfg_base: Config, x: float):
    """Evaluación completa (sizing + pesos + balance) para una x_LE."""
    cf = replace(cfg_base, x_le_ala=round(x, 3))
    rr = dimensionar(cfg=cf)
    mm, mt = desglose_oew(cf, rr)
    cx = _ctx(cf, rr, mm)
    res = evaluar_balance(cf, cx, cf.x_le_ala)
    marg = min(min(c["pct"] - cfg_base.pct_mac_min,
                   cfg_base.pct_mac_max - c["pct"]) for c in res["casos"])
    viol = max(c["viol"] for c in res["casos"])
    clave = ((1, round(marg, 6)) if viol <= 1e-9
             else (0, round(-viol, 6)))
    estado = (cf, rr, mm, mt, cx, res, marg, viol)
    return clave, estado


def buscar_posicion_ala(cfg: Config) -> tuple[float, tuple]:
    """
    Barrido de x_LE con evaluación completa en cada candidato (cada
    posición tiene sizing, colas por brazo, masas y balance propios).
    Elige la posición que hace cumplir los 3 casos en 15–35 % MAC con
    máximo margen; si ninguna los hace cumple, minimiza la violación.
    Devuelve (x_LE*, estado_en_x*).
    """
    def barrer(xs, mejor):
        for x in xs:
            clave, estado = _eval_x(cfg, x)
            if mejor is None or clave > mejor[0]:
                mejor = (clave, x, estado)
        return mejor

    mejor = None
    x = 8.0
    while x <= 16.01:  # refinado grueso: paso 0,4 m
        mejor = barrer([x], mejor)
        x += 0.4
    cx = mejor[1]
    mejor = barrer([cx + dx for dx in
                    [i * 0.08 for i in range(-5, 6)]], mejor)
    cx = mejor[1]
    mejor = barrer([cx + dx for dx in
                    [i * 0.02 for i in range(-4, 5)]], mejor)
    return mejor[1], mejor[2]


def dimensionar_con_balance(cfg: Config | None = None) -> dict:
    """
    Dimensionamiento + desglose de pesos (SUAVE) + balance en 3 casos +
    posición de ala iterada + bodega. Es la entrada que usan el reporte
    y el modelo 3D.
    """
    cfg = cfg or Config()
    x_base = cfg.x_le_ala

    # Estado de referencia en la posición inicial (para reportar el
    # movimiento y los CG previos)
    r0 = dimensionar(cfg=cfg)
    m0, meta0 = desglose_oew(cfg, r0)
    res0 = evaluar_balance(cfg, _ctx(cfg, r0, m0), x_base)

    x_star, estado = buscar_posicion_ala(cfg)
    cf, r, masas, meta, _, res, _, _ = estado

    if abs(x_star - x_base) < 0.02:
        cf, r, masas, meta = cfg, r0, m0, meta0
        res = res0

    items = {n: (masas[n], res["estaciones"][n]) for n in masas}

    r["x_le_ala"] = cf.x_le_ala
    r["x_le_ala_base"] = x_base
    r["mov_ala"] = cf.x_le_ala - x_base
    r["pesos_items"] = items
    r["pesos_meta"] = meta
    r["balance"] = res
    r["bodega"] = res["bodega"]
    r["balance_base"] = res0
    return r


# ---------------------------------------------------------------------------
# Geometría / dimensionamiento
# ---------------------------------------------------------------------------
def layout(cfg: Config) -> dict:
    """Layout longitudinal del fuselaje (no depende del MTOW)."""
    rows = math.ceil(cfg.n_pax / 4)
    l_cabina = rows * cfg.pitch + 0.30  # + fila extra sobre salidas
    x_cabina_fin = cfg.x_cabina_inicio + l_cabina
    x_tubo_fin = x_cabina_fin + 0.45
    l_fus = x_tubo_fin + cfg.long_cono

    # Ventanillas: una por fila, ambas costados (excluye fila de salida ala)
    pitch = cfg.pitch
    x_salida = cfg.x_le_ala + cfg.x_salida_off
    win = []
    for i in range(rows):
        x = cfg.x_cabina_inicio + 0.40 + i * pitch
        if abs(x - x_salida) < 0.45:
            continue  # fila ocupada por la salida de emergencia sobre el ala
        win.append(round(x, 2))

    return {
        "rows": rows,
        "l_cabina": l_cabina,
        "x_cabina_fin": x_cabina_fin,
        "x_tubo_fin": x_tubo_fin,
        "l_fus": l_fus,
        "x_puerta_delantera": cfg.x_puerta_delantera,
        "x_puerta_trasera": round(x_tubo_fin - 0.65, 2),
        "x_salida_ala": round(cfg.x_le_ala + cfg.x_salida_off, 2),
        "ventanillas_x": win,
    }


def dimensionar(mtow_inicial: float = 23000.0, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    lay = layout(cfg)

    mtow = mtow_inicial
    for _ in range(300):
        # ------------------------------------------------------------- ala
        s_ala = mtow / cfg.cargalar
        b = math.sqrt(cfg.aspecto * s_ala)
        c_root = 2 * s_ala / (b * (1 + cfg.lambda_wing))
        c_tip = cfg.lambda_wing * c_root
        mac = (2 / 3) * c_root * (
            (1 + cfg.lambda_wing + cfg.lambda_wing**2)
            / (1 + cfg.lambda_wing)
        )

        # Centro aerodinámico del ala y brazos de cola
        x_ac = (
            cfg.x_le_ala
            + math.tan(math.radians(cfg.barrido_le)) * (b / 2) / 2
            + mac / 4
        )
        x_le_ht = lay["x_tubo_fin"] + cfg.x_le_ht_off
        x_le_vt = lay["x_tubo_fin"] + cfg.x_le_vt_off

        # Colas por volúmenes: S_ht = v_h·S·mac / l_ht
        l_ht = x_le_ht - x_ac  # brazo (aprox.; error < 2 % por c_root/4)
        s_ht = cfg.v_h * s_ala * mac / (l_ht + 0.02 * mac)
        b_ht = math.sqrt(cfg.ar_ht * s_ht)

        l_vt = x_le_vt - x_ac
        s_vt = cfg.v_v * s_ala * b / (l_vt + 0.02 * b)
        b_vt = math.sqrt(cfg.ar_vt * s_vt)

        # ------------------------------------------------------ OEW por partes
        mojada = math.pi * cfg.d_fuselaje * lay["l_fus"] * 0.92
        comp = {
            "Ala (estructura + controles)": cfg.k_ala * s_ala,
            "Fuselaje presurizado": cfg.k_fuselaje * mojada,
            "Empennaje": cfg.k_empennaje * (s_ht + s_vt),
            "Tren retráctil": cfg.f_tren * mtow,
            "Grupo turbohélice (2×)": cfg.n_motores
            * (cfg.m_motor + cfg.m_helice + cfg.m_nacelle),
            "Sistemas y aviónica": cfg.m_sistemas,
            "Interior": cfg.kg_interior_pax * cfg.n_pax + cfg.m_interior_fijo,
            "Tripulación": tripulacion(cfg),
            "Varios": cfg.m_misc,
        }
        oew = sum(comp.values())

        comb = combustible(mtow, cfg)
        mtow_nuevo = oew + carga_util(cfg) + comb["total"]

        error = abs(mtow_nuevo - mtow)
        mtow += 0.6 * (mtow_nuevo - mtow)
        if error < 1.0:
            break

    # ---------------------------------------------------------- geometría final
    s_ala = mtow / cfg.cargalar
    b = math.sqrt(cfg.aspecto * s_ala)
    c_root = 2 * s_ala / (b * (1 + cfg.lambda_wing))
    c_tip = cfg.lambda_wing * c_root
    mac = (2 / 3) * c_root * (
        (1 + cfg.lambda_wing + cfg.lambda_wing**2) / (1 + cfg.lambda_wing)
    )
    x_ac = (
        cfg.x_le_ala
        + math.tan(math.radians(cfg.barrido_le)) * (b / 2) / 2
        + mac / 4
    )
    x_le_ht = lay["x_tubo_fin"] + cfg.x_le_ht_off
    x_le_vt = lay["x_tubo_fin"] + cfg.x_le_vt_off
    l_ht = x_le_ht - x_ac
    s_ht = cfg.v_h * s_ala * mac / (l_ht + 0.02 * mac)
    b_ht = math.sqrt(cfg.ar_ht * s_ht)
    l_vt = x_le_vt - x_ac
    s_vt = cfg.v_v * s_ala * b / (l_vt + 0.02 * b)
    b_vt = math.sqrt(cfg.ar_vt * s_vt)

    # ------------------------------------------------------------- motores
    comb = combustible(mtow, cfg)
    p_cruz_motor = comb["p_crucero_total"] / cfg.n_motores / 1000.0  # kW
    p_to_motor = math.ceil(p_cruz_motor / cfg.frac_cruz_to / 50.0) * 50.0
    t_estatico = cfg.k_empuje * p_to_motor * 1000.0  # N por motor
    masa_grupo = cfg.n_motores * (
        cfg.m_motor + cfg.m_helice + cfg.m_nacelle
    )

    # ----------------------------------------------------------- prestaciones
    mlw = mtow - comb["taxi"] - comb["crucero"]  # llega con reserva
    rho_sabe = RHO_SL * math.exp(-cfg.elev_sabe_m / 8435)
    rho_saar = RHO_SL * math.exp(-cfg.elev_saar_m / 8435)

    prestaciones = {}
    for nombre, rho in (("SABE", rho_sabe), ("SAAR", rho_saar)):
        vs_ld_mlw = math.sqrt(
            2 * (mlw * G) / (rho * s_ala * cfg.clmax_ld)
        )
        vs_ld_mtow = math.sqrt(
            2 * (mtow * G) / (rho * s_ala * cfg.clmax_ld)
        )
        v_app = 1.3 * vs_ld_mlw
        tofl, tofl_ground, tofl_oei = distancia_despegue(
            mlw_p=mtow,  # despegue al MTOW
            cfg=cfg, s=s_ala, rho=rho,
            p_to_motor=p_to_motor,
            t_estatico_motor=t_estatico,
        )
        ld_ground, ld_air, ld_total = distancia_aterrizaje(
            mlw, cfg, s_ala, rho, v_app,
        )
        prestaciones[nombre] = {
            "rho": rho,
            "vs_ld_mlw": vs_ld_mlw,
            "vs_ld_mtow": vs_ld_mtow,
            "v_app": v_app,
            "tofl": tofl,
            "tofl_ground": tofl_ground,
            "tofl_oei": tofl_oei,
            "ld_ground": ld_ground,
            "ld_air": ld_air,
            "ld_total": ld_total,
        }

    oew_final = mtow - carga_util(cfg) - comb["total"]

    r = {
        "mtow": mtow,
        "mlw": mlw,
        "oew": oew_final,
        "x_le_ala": cfg.x_le_ala,
        "payload": carga_util(cfg),
        "comb_taxi": comb["taxi"],
        "comb_crucero": comb["crucero"],
        "comb_espera": comb["espera"],
        "comb_desvio": comb["desvio"],
        "comb_total": comb["total"],
        "ff_crucero": comb["ff_crucero"],
        "area_ala": s_ala,
        "envergadura": b,
        "c_root": c_root,
        "c_tip": c_tip,
        "mac": mac,
        "aspecto": cfg.aspecto,
        "x_ac": x_ac,
        "s_ht": s_ht,
        "b_ht": b_ht,
        "c_root_ht": 2 * s_ht / (b_ht * (1 + cfg.lambda_ht)),
        "s_vt": s_vt,
        "b_vt": b_vt,
        "c_root_vt": 2 * s_vt / (b_vt * (1 + cfg.lambda_vt)),
        "l_ht": x_le_ht - x_ac,
        "l_vt": x_le_vt - x_ac,
        "x_le_ht": x_le_ht,
        "x_le_vt": x_le_vt,
        "p_cruz_motor": p_cruz_motor,
        "p_to_motor": p_to_motor,
        "t_estatico_motor": t_estatico,
        "masa_grupo": masa_grupo,
        "prestaciones": prestaciones,
    }
    r.update(lay)
    return r


# ---------------------------------------------------------------------------
# Prestaciones: despegue y aterrizaje
# ---------------------------------------------------------------------------
def _cd_inducido(cl: float, ar: float, e: float) -> float:
    return cl**2 / (math.pi * ar * e)


def _rodaje(cfg, s, rho, w, cl_g, cd0, fraccion_potencia, vs,
            p_to_total, t_estatico_total, v_fin=None):
    """Integra la carrera de rodaje. Devuelve (distancia, V_LOF)."""
    v_lof = 1.15 * vs
    v_fin = v_fin if v_fin is not None else v_lof
    t_stat = fraccion_potencia * t_estatico_total
    p = fraccion_potencia * p_to_total
    dist = 0.0
    dv = 0.5
    v = 0.1
    while v < v_fin:
        t = min(t_stat, cfg.eta_prop * p / max(v, 1.0))
        q = 0.5 * rho * v * v
        l = min(q * s * cl_g, w * 1.05)
        d = q * s * (cd0 + _cd_inducido(cl_g, cfg.aspecto, cfg.oswald))
        a = G * ((t - d) / w - cfg.mu_to * max(1.0 - l / w, 0.0))
        a = max(0.05, a)
        dist += v * dv / a
        v += dv
    return dist, v_lof


def distancia_despegue(mlw_p: float, cfg: Config, s: float, rho: float,
                       p_to_motor: float, t_estatico_motor: float):
    """
    Distancia de despegue (m). Devuelve
    (total potencia total, rodaje, total motor inoperante).

    Integración de la carrera con T = min(Testática, η·P/V) y arrastre con
    flaps de despegue; salida hasta 15 m. El caso motor inoperante integra
    con media potencia y agrega el tramo de ascenso con un motor a
    V2 = 1.2·VS_TO.
    """
    w = mlw_p * G
    vs_to = math.sqrt(2 * w / (rho * s * cfg.clmax_to))
    v2 = 1.2 * vs_to
    cd0_to = cfg.cd0_limpio + cfg.dcd_tren + cfg.dcd_flap_to
    cl_g = 0.7 * cfg.clmax_to  # ala con flaps, ángulo nulo durante rodaje

    cfg_n = cfg.n_motores
    p_to_total = cfg_n * p_to_motor * 1000.0
    t_est_total = cfg_n * t_estatico_motor
    v_lof = 1.15 * vs_to

    # --- Potencia total (FAR 25.113(a): a 15 m con todos los motores)
    roll, _ = _rodaje(cfg, s, rho, w, cl_g, cd0_to, 1.0, vs_to,
                      p_to_total, t_est_total)
    tofl = roll * 1.6  # rodaje ≈ 62 % de la distancia total a 15 m

    # --- Motor inoperante (FAR 25.113(b)): falla en V1 ≈ 0,96·V_LOF con
    #     todos los motores, seguido de aceleración a V2 y ascenso a 15 m
    #     con un motor; se toma el mayor que 1,15 × distancia con todos.
    v1 = 0.96 * v_lof
    roll_v1, _ = _rodaje(cfg, s, rho, w, cl_g, cd0_to, 1.0, vs_to,
                         p_to_total, t_est_total, v_fin=v1)
    t1v = min(t_est_total * 0.5, cfg.eta_prop * p_to_total * 0.5 / v1)
    q1 = 0.5 * rho * v1 * v1
    l1 = min(q1 * s * cl_g, w * 1.05)
    d1 = q1 * s * (cd0_to + _cd_inducido(cl_g, cfg.aspecto, cfg.oswald))
    a1 = max(G * ((t1v - d1) / w - cfg.mu_to * max(1.0 - l1 / w, 0.0)), 0.1)
    s_v1_vlof = (v_lof**2 - v1**2) / (2 * a1)

    q = 0.5 * rho * v2 * v2
    cl2 = 2 * w / (rho * s * v2**2)
    cd2 = cd0_to + _cd_inducido(cl2, cfg.aspecto, cfg.oswald)
    d2 = q * s * cd2
    t1 = min(
        t_est_total * 0.5,
        cfg.eta_prop * p_to_total * 0.5 / v2,
    )
    sin_gamma = (t1 - d2) / w
    sin_gamma = max(sin_gamma, 0.015)
    climb = 15.0 / sin_gamma  # hasta 15 m
    oei_acepta = roll_v1 + s_v1_vlof + 120.0 + climb
    tofl_oei = max(1.15 * tofl, oei_acepta)

    return tofl, roll, tofl_oei


def distancia_aterrizaje(mlw: float, cfg: Config, s: float, rho: float,
                         v_app: float):
    """Distancia total de aterrizaje (m): planada + rodaje con frenos."""
    w = mlw * G
    v_td = v_app  # conservador: toque a V_APP
    cl_ld = 2 * w / (rho * s * v_td**2)
    q = 0.5 * rho * v_td**2
    cd_ld = (cfg.cd0_limpio + cfg.dcd_tren + cfg.dcd_flap_ld
             + _cd_inducido(cl_ld, cfg.aspecto, cfg.oswald))
    l_w = min(q * s * cl_ld / w, 0.99)
    d_w = q * s * cd_ld / w
    a = G * (cfg.mu_frenado * (1 - l_w) + cfg.f_reversa + d_w)
    ground = v_td**2 / (2 * a)
    air = v_app * 7.5  # planada desde ~15 m (pantalla de 50 ft)
    return ground, air, ground + air


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def imprimir_reporte(cfg: Config, r: dict) -> None:
    linea = "=" * 72
    print(linea)
    print(" DIMENSIONAMIENTO — Avión regional turbohélice (30–90 pax)")
    print(" Ruta de diseño: SABE ⇄ SAAR · punto de diseño 72 pax (2+2)")
    print(linea)

    print("\nRuta y misión")
    print(f"  Buenos Aires SABE: pista 13/31, {cfg.pista_sabe_m:.0f} m, "
          f"elev. {cfg.elev_sabe_m:.0f} m")
    print(f"  Rosario      SAAR: pista 02/20, {cfg.pista_saar_m:.0f} m, "
          f"elev. {cfg.elev_saar_m:.0f} m")
    print(f"  Distancia de ruta: {cfg.distancia_ruta_km:.0f} km · "
          f"crucero FL{cfg.fl_cruz_m / 30.48:.0f} a "
          f"{cfg.v_cruz_kmh:.0f} km/h TAS")
    print(f"  Reserva: {cfg.espera_h * 60:.0f} min de espera + "
          f"{cfg.div_km:.0f} km de desvío")

    print("\nCargas")
    print(f"  Pasajeros ({cfg.n_pax} × {cfg.masa_pax:.0f} kg con equipaje) "
          f"  {r['payload']:8.1f} kg")
    print(f"  Tripulación de vuelo/cabina          {tripulacion(cfg):8.1f} kg")

    print("\nDesglose de pesos por componente")
    items = r.get("pesos_items")
    meta = r.get("pesos_meta") or {}
    if items:
        print(f"  Método: {meta.get('fuente', 'n/d')}")
        if meta.get("suave_raw"):
            print(f"  Referencia SUAVE sin escala: {meta['suave_raw']:.1f} kg "
                  f"· factor de escala de estructura ×{meta['k_escala']:.2f} "
                  f"({meta.get('nota_k', '')})")
        for grupo, nombres in GRUPOS_ITEMS:
            print(f"  {grupo}:")
            sub = 0.0
            for n in nombres:
                kg, x = items[n]
                sub += kg
                print(f"    {n:<30} {kg:8.1f} kg   x = {x:5.2f} m")
            print(f"    {'Subtotal ' + grupo.lower():<30} {sub:8.1f} kg")
        oew_sum = sum(m for m, _ in items.values())
        print("  " + "—" * 50)
        print(f"  {'Peso en vacío operativo (OEW)':<30} "
              f"{oew_sum:8.1f} kg ({100 * oew_sum / r['mtow']:.1f} % MTOW)")
    else:
        print(f"  Peso en vacío operativo (OEW)       {r['oew']:8.1f} kg "
              f"({100 * r['oew'] / r['mtow']:.1f} % del MTOW)")
    print(f"  {'Combustible taxi + ascenso':<30} {r['comb_taxi']:8.1f} kg")
    print(f"  {'Combustible de crucero':<30} {r['comb_crucero']:8.1f} kg")
    print(f"  {'Reserva: espera + desvío':<30} "
          f"{r['comb_espera'] + r['comb_desvio']:8.1f} kg")
    print(f"  {'Combustible total (tanque lleno)':<30} "
          f"{r['comb_total']:8.1f} kg")
    print(f"  {'MTOW (punto de diseño 72 pax)':<30} {r['mtow']:8.1f} kg")
    print(f"  {'MLW (peso máx. de aterrizaje)':<30} {r['mlw']:8.1f} kg")
    if items:
        n = cfg.n_pax_balance
        tot3 = (r["oew"] + r["comb_total"]
                + n * (cfg.kg_pax_solo + cfg.kg_equipaje_pax))
        print(f"  {f'Carga caso 3 ({n} pax)':<30} "
              f"{n * (cfg.kg_pax_solo + cfg.kg_equipaje_pax):8.1f} kg")
        nota = ""
        if tot3 > r["mtow"]:
            nota = (f"  ← excede MTOW en {tot3 - r['mtow']:.0f} kg "
                    f"(envolvente de balance, no operable)")
        print(f"  TOTAL caso 3 de carga (OEW+comb.+carga) "
              f"{tot3:8.1f} kg{nota}")

    print("\nDimensiones")
    print(f"  Superficie alar (S)                 {r['area_ala']:8.1f} m²")
    print(f"  Envergadura (b)                     {r['envergadura']:8.2f} m")
    print(f"  Longitud total (L)                  {r['l_fus']:8.2f} m")
    print(f"  Cuerda raíz / media (c_r / MAC)     "
          f"{r['c_root']:.2f} / {r['mac']:.2f} m")
    print(f"  Relación de aspecto                 {r['aspecto']:8.1f}")
    print(f"  Carga alar al MTOW                  "
          f"{r['mtow'] / r['area_ala']:8.1f} kg/m²")
    print(f"  Fuselaje: Ø {cfg.d_fuselaje:.2f} m · cabina "
          f"{r['rows']} filas 2+2 · {len(r['ventanillas_x'])} ventanillas/lado")
    print(f"  Estab. horizontal: S = {r['s_ht']:.1f} m², "
          f"b = {r['b_ht']:.1f} m (V_h = {cfg.v_h:.2f})")
    print(f"  Derivada vertical: S = {r['s_vt']:.1f} m², "
          f"h = {r['b_vt']:.1f} m (V_v = {cfg.v_v:.3f})")

    print("\nPropulsión (2 turbohélices en el ala)")
    print(f"  Potencia de despegue por motor      {r['p_to_motor']:.0f} kW "
          f"({r['p_to_motor'] * 1.341:.0f} shp)")
    print(f"  Empuje estático por motor           "
          f"{r['t_estatico_motor'] / 1000:.1f} kN")
    print(f"  Potencia de crucero por motor       {r['p_cruz_motor']:.0f} kW "
          f"(caudal total {r['ff_crucero']:.0f} kg/h)")
    print(f"  Hélices de {cfg.n_palas} palas, Ø {cfg.diam_helice:.2f} m")

    p = r["prestaciones"]
    vs = p["SABE"]["vs_ld_mlw"]
    print("\nVelocidades de pérdida (con slats + flaps completos, MLW)")
    print(f"  V_S (flaps)                         {vs:8.1f} m/s = "
          f"{vs * 1.94384:5.1f} kt")
    print(f"  V_S al MTOW                         "
          f"{p['SABE']['vs_ld_mtow'] * 1.94384:8.1f} kt")
    print(f"  V_APP (1,3·V_S)                     "
          f"{p['SABE']['v_app'] * 1.94384:8.1f} kt")

    def fila(nombre, dist, pista):
        margen = 100 * (pista - dist) / pista
        ok = "OK " if margen >= 15 else ("¡REVISAR!" if margen < 0 else "OK")
        return (f"  {nombre:<28} {dist:7.0f} m  vs {pista:5.0f} m  "
                f"margen {margen:5.1f} %  [{ok}]")

    print("\nDistancias estimadas (ISA, pista seca, MTOW en despegue / "
          "MLW en aterrizaje)")
    print(f"  {'DESPEGUE — potencia total':<28} {p['SABE']['tofl']:7.0f} m "
          f"(rodaje {p['SABE']['tofl_ground']:.0f} m)  |  "
          f"SAAR {p['SAAR']['tofl']:7.0f} m")
    print(f"  {'DESPEGUE — 1 motor inoperante':<28} "
          f"{p['SABE']['tofl_oei']:7.0f} m  |  "
          f"SAAR {p['SAAR']['tofl_oei']:7.0f} m   ← caso gobernante")
    print(f"  {'ATERRIZAJE — total':<28} {p['SABE']['ld_total']:7.0f} m "
          f"(rodaje {p['SABE']['ld_ground']:.0f} m + planada "
          f"{p['SABE']['ld_air']:.0f} m)  |  SAAR "
          f"{p['SAAR']['ld_total']:7.0f} m")
    print("\n  Verificación contra pistas (caso gobernante):")
    print(fila("Despegue SABE 13/31 (motor inop.)",
               p["SABE"]["tofl_oei"], cfg.pista_sabe_m))
    print(fila("Despegue SAAR 02/20 (motor inop.)",
               p["SAAR"]["tofl_oei"], cfg.pista_saar_m))
    print(fila("Aterrizaje SABE 13/31",
               p["SABE"]["ld_total"], cfg.pista_sabe_m))
    print(fila("Aterrizaje SAAR 02/20",
               p["SAAR"]["ld_total"], cfg.pista_saar_m))

    print("\nSuperficies de control y alto sustentación")
    print("  · Slats completos + flaps 35° (V_S ≈ 90 kt en MLW)")
    print("  · Slats + flaps 15° para despegue (CLmax_TO = 2,10)")
    print("  · Ailerones en punta · spoilers de frenado/descompresión "
          "en el ala")
    print("  · Timón de profundidad (HT) y timón de dirección (VT), "
          "asistidos")

    bal = r.get("balance")
    if bal:
        mac = r["mac"]
        x_le_mac = bal["x_mac_le"]
        print("\nCentro de gravedad — 3 casos de carga")
        print(f"  Rango aceptable: {cfg.pct_mac_min:.0f}–"
              f"{cfg.pct_mac_max:.0f} % de la MAC ({mac:.2f} m; ventana "
              f"de {0.20 * mac:.2f} m)")
        print(f"  Cuerda media: x = {x_le_mac:.2f} m (borde de ataque) "
              f"→ {x_le_mac + mac:.2f} m (borde de fuga)")
        print(f"  {'Caso':<46} {'Peso':>8} {'x_CG':>7} {'%MAC':>7}  Estado")
        for c in bal["casos"]:
            estado = "OK" if c["viol"] <= 1e-9 else "FUERA DE RANGO"
            print(f"  {c['nombre']:<46} {c['peso']:8.0f} kg "
                  f"{c['x_cg']:7.2f} {c['pct']:7.1f}  [{estado}]")
        fuera = [c for c in bal["casos"] if c["viol"] > 1e-9]
        if fuera:
            for c in fuera:
                lado = ("delantero" if c["pct"] < cfg.pct_mac_min
                        else "trasero")
                print(f"  ¡Atención! caso «{c['nombre']}» queda fuera "
                      f"({c['pct']:.1f} % MAC, límite {lado}); posición "
                      f"óptima del ala alcanzada en x_LE = "
                      f"{r['x_le_ala']:.2f} m.")
        b0 = r.get("balance_base")
        if b0:
            p0 = " / ".join(f"{c['pct']:.1f}" for c in b0["casos"])
            fuera0 = any(c["viol"] > 1e-9 for c in b0["casos"])
            print(f"  Posición inicial (x_LE = {r['x_le_ala_base']:.2f} m): "
                  f"CG = {p0} % MAC"
                  + (" — los 3 casos FUERA de rango" if fuera0
                     else " — ya cumplían"))
        mov = r.get("mov_ala", 0.0)
        if abs(mov) < 0.02:
            print(f"  Posición del ala: sin cambios "
                  f"(x_LE = {r['x_le_ala']:.2f} m ya cumple en los 3 casos)")
        else:
            print(f"  Posición del ala ajustada: x_LE "
                  f"{r['x_le_ala_base']:.2f} → {r['x_le_ala']:.2f} m "
                  f"(Δ = {mov:+.2f} m); motores, tren y salidas sobre el "
                  f"ala se desplazaron con el ala")
        print("  Nota: los casos 2 y 3 se evalúan sin combustible / con "
              "tanque lleno según la definición de cada caso; el caso 3 "
              "es envolvente de balance (no requiere operar a ese peso).")

    bod = r.get("bodega")
    if bod:
        n = cfg.n_pax_balance
        print("\nBodega de equipaje (bajo el piso de la cabina)")
        print(f"  Volumen requerido: {bod['V']:.1f} m³ "
              f"({n} pax × {cfg.vol_equipaje_pax:.2f} m³/pax) · masa máx. "
              f"{bod['masa_max']:.0f} kg (15 kg/pax)")
        print(f"  Sección útil bajo el piso: {bod['A_seg']:.2f} m² "
              f"(Ø {cfg.d_fuselaje:.2f} m, piso z = {bod['z0']:.2f} m) × "
              f"útil {cfg.fr_util_bodega * 100:.0f} % = {bod['A_us']:.2f} m²")
        print(f"  Posición: x = {bod['x0']:.2f} → {bod['x1']:.2f} m "
              f"(centro {bod['centro']:.2f} m), z = {bod['z1']:.2f} → "
              f"{bod['z0']:.2f} m · longitud {bod['L']:.2f} m")
        if bal:
            cg3 = bal["casos"][2]["x_cg"]
            print(f"  Brazo respecto al CG del caso 3 (x = {cg3:.2f} m): "
                  f"{bod['centro'] - cg3:+.2f} m — minimiza el efecto de "
                  f"cargar/descargar equipaje")
    print(linea)


def main() -> None:
    cfg = Config()
    r = dimensionar_con_balance(cfg=cfg)
    imprimir_reporte(cfg, r)


if __name__ == "__main__":
    main()
