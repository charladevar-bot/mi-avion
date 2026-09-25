#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimación preliminar (método clase I) del peso y las dimensiones
aproximadas de un avión liviano de 2 pasajeros.

Metodo
------
* Carga útil: pasajeros + equipaje.
* Combustible: Breguet para hélice en crucero + fracción de taxi/ascenso
  + reserva de 45 min al 65 % de la potencia de crucero.
* Peso en vacío operativo (OEW): suma de componentes con densidades
  estructurales estadísticas típicas de un avión ligero (ala arriostrada,
  tren fijo, grupo motriz de émbolo tipo Rotax).
* Iteración de punto fijo: el MTOW determina el área alar y a los
  componentes que dependen del peso, y éstos a su vez determinan el MTOW,
  hasta converger.
* Dimensiones: área alar y envergadura de los elegidos, más reglas
  empíricas de fuselaje y volúmenes de cola (V_h, V_v).

Sólo usa la biblioteca estándar. Las librerías de requirements.txt
(aerosandbox, openmdao, openaerostruct, openconcept) quedan instaladas en
.venv para las etapas siguientes: aerodinámica, estructuras y optimización.

Uso:  .venv/bin/python sizing_2pax.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.80665  # m/s²
RHO_AVGAS = 0.72  # kg/L


# ---------------------------------------------------------------------------
# Definición de configuración (ajustable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Cargas útiles ---
    n_pasajeros: int = 2
    masa_pasajero: float = 80.0  # kg por pasajero
    equipaje: float = 20.0  # kg total

    # --- Misión ---
    alcance_km: float = 650.0  # crucero a nivel del mar ISA
    v_crucero_kmh: float = 190.0
    reserva_h: float = 0.75  # 45 min
    reserva_potencia: float = 0.65  # fracción de potencia de crucero
    taxi_ascenso_mtow: float = 0.015  # taxi + ascenso inicial (% del MTOW)

    # --- Propulsión y aerodinámica ---
    bsfc: float = 0.30  # kg/(kW·h), clase Rotax 912
    eta_helice: float = 0.80
    l_d_crucero: float = 9.5  # L/D máximo de crucero

    # --- Dimensionamiento de ala ---
    cargalar: float = 52.0  # kg/m² (W/S al MTOW)
    aspecto: float = 8.0  # relación de aspecto del ala

    # --- Densidades estructurales (kg/m²) ---
    k_ala: float = 8.5  # ala arriostrada
    k_fuselaje: float = 4.0  # por área mojada
    k_ht: float = 7.5
    k_vt: float = 7.0

    # --- Fracciones y masas fijas (kg) ---
    f_tren: float = 0.06  # tren fijo de nariz: f_tren × MTOW
    masa_grupo_motriz: float = 80.0  # motor + hélice + montaje + escapes
    masa_sistemas: float = 45.0  # eléctrico, instrumentos, aviónica
    masa_interior: float = 30.0  # asientos, cinturones, guarnecido
    masa_varios: float = 5.0  # fluidos, aceite, herramientas

    # --- Volúmenes de cola ---
    v_h: float = 0.55  # coeficiente de volumen horizontal
    v_v: float = 0.05  # coeficiente de volumen vertical
    ar_ht: float = 4.5
    ar_vt: float = 1.3

    # --- Geometría del fuselaje (m) ---
    l_nariz: float = 1.20  # disco de hélice -> cortafuegos (motor de émbolo)
    l_cabina: float = 2.10  # cabina de 2 en fila
    l_cola: float = 3.70  # fuselaje de cola
    ancho_fus: float = 1.15
    alto_fus: float = 1.30
    f_mojado: float = 0.85  # descuento por ventanas, entradas, etc.

    # Brazos de cola (fracción de la longitud total)
    arm_ht: float = 0.60
    arm_vt: float = 0.55


# ---------------------------------------------------------------------------
# Funciones de estimación
# ---------------------------------------------------------------------------
def carga_util(cfg: Config) -> float:
    """Carga útil total (kg)."""
    return cfg.n_pasajeros * cfg.masa_pasajero + cfg.equipaje


def combustible(mtow: float, cfg: Config) -> tuple[float, float, float]:
    """
    Combustible total (kg) desglosado en (taxi+ascenso, crucero, reserva).

    Breguet para hélice en crucero:
        R = (eta·L/D) / (c'·g) · ln(W1/W2)   con c' en kg/J
    """
    # Taxi + ascenso
    w_taxi = cfg.taxi_ascenso_mtow * mtow
    w_ini_crucero = mtow - w_taxi

    # Factor de alcance: k = (eta·L/D)/(c'·g)  [m]
    c_prime = cfg.bsfc / 3.6e6  # kg/J
    k = (cfg.eta_helice * cfg.l_d_crucero) / (c_prime * G)
    w_fin_crucero = w_ini_crucero * math.exp(-cfg.alcance_km * 1e3 / k)
    w_crucero = w_ini_crucero - w_fin_crucero

    # Reserva: 45 min al 65 % de la potencia de crucero (peso medio)
    v = cfg.v_crucero_kmh / 3.6  # m/s
    w_medio = 0.5 * (w_ini_crucero + w_fin_crucero)
    p_traccion = w_medio * G * v / cfg.l_d_crucero  # W
    p_eje = p_traccion / cfg.eta_helice  # W
    flujo = cfg.bsfc * (p_eje / 1000.0)  # kg/h
    w_reserva = flujo * cfg.reserva_h * cfg.reserva_potencia

    return w_taxi, w_crucero, w_reserva


def area_mojada_fuselaje(cfg: Config) -> tuple[float, float, float]:
    """(longitud, área mojada descontada, área frontal-equivalente) del fuselaje."""
    l_fus = cfg.l_nariz + cfg.l_cabina + cfg.l_cola
    a, b = cfg.ancho_fus / 2.0, cfg.alto_fus / 2.0
    # Perímetro elíptico (fórmula de Ramanujan) × longitud
    per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return l_fus, per * l_fus * cfg.f_mojado, math.pi * a * b


def oew_estimado(
    mtow: float, area_ala: float, envergadura: float, mac: float,
    l_fus: float, area_mojada: float, cfg: Config,
) -> tuple[float, dict[str, float]]:
    """Peso en vacío operativo (kg) y desglose por componente."""
    # Colas por volúmenes
    s_ht = cfg.v_h * mac * area_ala / (cfg.arm_ht * l_fus)
    s_vt = cfg.v_v * envergadura * area_ala / (cfg.arm_vt * l_fus)

    comp = {
        "Ala": cfg.k_ala * area_ala,
        "Fuselaje": cfg.k_fuselaje * area_mojada,
        "Estab. horizontal": cfg.k_ht * s_ht,
        "Derivada": cfg.k_vt * s_vt,
        "Tren de aterrizaje": cfg.f_tren * mtow,
        "Grupo motriz": cfg.masa_grupo_motriz,
        "Sistemas y aviónica": cfg.masa_sistemas,
        "Interior": cfg.masa_interior,
        "Varios": cfg.masa_varios,
    }
    return sum(comp.values()), comp


def dimensionar(mtow: float, cfg: Config) -> dict[str, float]:
    """Iteración de punto fijo hasta converger MTOW y obtener geometría."""
    mtow_actual = mtow
    n_iter = 0
    for _ in range(500):
        n_iter += 1
        # Geometría derivada del MTOW actual
        area_ala = mtow_actual / cfg.cargalar
        envergadura = math.sqrt(area_ala * cfg.aspecto)
        mac = area_ala / envergadura
        l_fus, mojada, _ = area_mojada_fuselaje(cfg)
        oew, comp = oew_estimado(
            mtow_actual, area_ala, envergadura, mac, l_fus, mojada, cfg
        )
        wt_taxi, wt_cru, wt_res = combustible(mtow_actual, cfg)
        mtow_nuevo = oew + carga_util(cfg) + wt_taxi + wt_cru + wt_res

        error = abs(mtow_nuevo - mtow_actual)
        mtow_actual += 0.6 * (mtow_nuevo - mtow_actual)  # relajación
        if error < 0.05:
            break

    # Geometría final con el MTOW convergido
    area_ala = mtow_actual / cfg.cargalar
    envergadura = math.sqrt(area_ala * cfg.aspecto)
    mac = area_ala / envergadura
    l_fus, mojada, _ = area_mojada_fuselaje(cfg)
    oew, comp = oew_estimado(
        mtow_actual, area_ala, envergadura, mac, l_fus, mojada, cfg
    )
    wt_taxi, wt_cru, wt_res = combustible(mtow_actual, cfg)

    s_ht = cfg.v_h * mac * area_ala / (cfg.arm_ht * l_fus)
    s_vt = cfg.v_v * envergadura * area_ala / (cfg.arm_vt * l_fus)

    return {
        "mtow": mtow_actual,
        "oew": oew,
        "carga_util": carga_util(cfg),
        "comb_taxi": wt_taxi,
        "comb_crucero": wt_cru,
        "comb_reserva": wt_res,
        "comb_total": wt_taxi + wt_cru + wt_res,
        "area_ala": area_ala,
        "envergadura": envergadura,
        "mac": mac,
        "l_fus": l_fus,
        "ancho_fus": cfg.ancho_fus,
        "alto_fus": cfg.alto_fus,
        "mojada": mojada,
        "s_ht": s_ht,
        "s_vt": s_vt,
        "b_ht": math.sqrt(cfg.ar_ht * s_ht),
        "b_vt": math.sqrt(cfg.ar_vt * s_vt),
        "cargalar_real": mtow_actual / area_ala,
        "iteraciones": n_iter,
        "componentes": comp,
    }


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def imprimir_reporte(cfg: Config, r: dict[str, float]) -> None:
    linea = "=" * 66
    print(linea)
    print(" ESTIMACIÓN PRELIMINAR — Avión liviano de 2 pasajeros")
    print(linea)

    print("\nMisión y cargas útiles")
    print(f"  Pasajeros ({cfg.n_pasajeros} × {cfg.masa_pasajero:.0f} kg) "
          f"      {cfg.n_pasajeros * cfg.masa_pasajero:6.1f} kg")
    print(f"  Equipaje                     {cfg.equipaje:6.1f} kg")
    print(f"  Carga útil total             {r['carga_util']:6.1f} kg")
    print(f"  Alcance (crucero, nivel del mar) {cfg.alcance_km:6.0f} km")
    print(f"  Velocidad de crucero         {cfg.v_crucero_kmh:6.0f} km/h")
    print(f"  Reserva                      {cfg.reserva_h * 60:6.0f} min "
          f"al {cfg.reserva_potencia * 100:.0f} % de potencia")

    print("\nPeso")
    print("  Desglose OEW (estadístico):")
    for k, v in r["componentes"].items():
        print(f"    {k:<24} {v:7.1f} kg")
    print(f"  {'—'*36}")
    print(f"  Peso vacío operativo (OEW)   {r['oew']:7.1f} kg")
    print(f"  Combustible taxi + ascenso   {r['comb_taxi']:7.1f} kg")
    print(f"  Combustible de crucero       {r['comb_crucero']:7.1f} kg")
    print(f"  Combustible de reserva       {r['comb_reserva']:7.1f} kg")
    print(f"  Combustible total            {r['comb_total']:7.1f} kg "
          f"(≈ {r['comb_total'] / RHO_AVGAS:.0f} L de avgas)")
    print(f"  Peso máx. al despegue (MTOW) {r['mtow']:7.1f} kg")
    print(f"  Relación OEW/MTOW            {100 * r['oew'] / r['mtow']:7.1f} %")
    print(f"  Fracción de combustible      "
          f"{100 * r['comb_total'] / r['mtow']:7.1f} %")

    print("\nSuperficies y dimensiones aproximadas")
    print(f"  Área alar                    {r['area_ala']:7.1f} m²")
    print(f"  Envergadura                  {r['envergadura']:7.2f} m")
    print(f"  Cuerda media (MAC)           {r['mac']:7.2f} m")
    print(f"  Relación de aspecto          {cfg.aspecto:7.1f}")
    print(f"  Carga alar (W/S)             {r['cargalar_real']:7.1f} kg/m²")
    print(f"  Longitud total fuselaje      {r['l_fus']:7.2f} m")
    print(f"  Ancho fuselaje               {r['ancho_fus']:7.2f} m")
    print(f"  Alto fuselaje                {r['alto_fus']:7.2f} m")
    print(f"  Área mojada fuselaje         {r['mojada']:7.1f} m²")
    print(f"  Estab. horizontal (S/b)      {r['s_ht']:7.1f} m² / "
          f"{r['b_ht']:.1f} m")
    print(f"  Derivada (S/b)               {r['s_vt']:7.1f} m² / "
          f"{r['b_vt']:.1f} m")

    print("\nReferencias de clase")
    print("  Cessna 152 (2 plazas):  526 kg MTOW | 10.06 m envergadura | "
          "7.65 m largo | 14.9 m² área alar")
    print("  Clase LSA (2 plazas):   MTOW máx. 600 kg (sin paracaídas)")

    print("\nSupuestos clave")
    print(f"  · {cfg.masa_pasajero:.0f} kg por pasajero, "
          f"{cfg.equipaje:.0f} kg de equipaje")
    print(f"  · L/D crucero = {cfg.l_d_crucero}, "
          f"η hélice = {cfg.eta_helice}, BSFC = {cfg.bsfc} kg/(kW·h)")
    print(f"  · Ala arriostrada (k = {cfg.k_ala} kg/m²), tren fijo "
          f"({cfg.f_tren * 100:.0f} % del MTOW)")
    print(f"  · Convergencia en {r['iteraciones']} iteraciones")
    print(linea)


def main() -> None:
    cfg = Config()
    resultado = dimensionar(mtow=600.0, cfg=cfg)
    imprimir_reporte(cfg, resultado)


if __name__ == "__main__":
    main()
