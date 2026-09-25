#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Uso:  .venv/bin/python sizing_regional.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    x_salida_ala: float = 9.60
    long_cono: float = 4.80
    x_le_ht_off: float = 0.15  # desde fin de tubo
    x_le_vt_off: float = 1.20  # desde fin de tubo

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
    win = []
    for i in range(rows):
        x = cfg.x_cabina_inicio + 0.40 + i * pitch
        if abs(x - cfg.x_salida_ala) < 0.45:
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
        "x_salida_ala": cfg.x_salida_ala,
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
        if a <= 0.05:
            a = 0.05
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

    print("\nPesos")
    print("  Desglose OEW:")
    s_ala = r["area_ala"]
    mojada = math.pi * cfg.d_fuselaje * r["l_fus"] * 0.92
    comp = {
        "Ala (estructura + controles)": cfg.k_ala * s_ala,
        "Fuselaje presurizado": cfg.k_fuselaje * mojada,
        "Empennaje": cfg.k_empennaje * (r["s_ht"] + r["s_vt"]),
        "Tren retráctil": cfg.f_tren * r["mtow"],
        "Grupo turbohélice (2×)": r["masa_grupo"],
        "Sistemas y aviónica": cfg.m_sistemas,
        "Interior": cfg.kg_interior_pax * cfg.n_pax + cfg.m_interior_fijo,
        "Tripulación": tripulacion(cfg),
        "Varios": cfg.m_misc,
    }
    for k, v in comp.items():
        print(f"    {k:<36} {v:8.1f} kg")
    print("  " + "—" * 44)
    print(f"  Peso en vacío operativo (OEW)       {r['oew']:8.1f} kg "
          f"({100 * r['oew'] / r['mtow']:.1f} % del MTOW)")
    print(f"  Combustible taxi + ascenso          {r['comb_taxi']:8.1f} kg")
    print(f"  Combustible de crucero              {r['comb_crucero']:8.1f} kg")
    print(f"  Reserva: espera + desvío            "
          f"{r['comb_espera'] + r['comb_desvio']:8.1f} kg")
    print(f"  Combustible total                   {r['comb_total']:8.1f} kg")
    print(f"  MTOW                                {r['mtow']:8.1f} kg")
    print(f"  MLW (peso máx. de aterrizaje)       {r['mlw']:8.1f} kg")

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
    print(linea)


def main() -> None:
    cfg = Config()
    r = dimensionar(cfg=cfg)
    imprimir_reporte(cfg, r)


if __name__ == "__main__":
    main()
