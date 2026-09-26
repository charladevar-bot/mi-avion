#!/usr/bin/env python3
"""
Integración con SUAVE 2.5.2 (suavecode/SUAVE, instalado en editable desde
GitHub) para el desglose de pesos por componente.

SUAVE fue instalado con:  pip install --no-build-isolation -e suave/trunk
(documentación del repo: git clone + cd trunk + setup.py install; se usó
el equivalente moderno pip -e).

Este módulo:
  1. Aplica shims de compatibilidad (SUAVE 2.5.2 fue escrito para
     scipy/numpy antiguos: `cumtrapz` desapareció en scipy ≥ 1.12).
  2. Construye un `vehicle` SUAVE mínimo a partir del dimensionamiento.
  3. Llama a SUAVE.Methods.Weights.Correlations.General_Aviation.empty,
     que aplica correlaciones por componente (fuselaje, alas, colas,
     tren, sistemas — con desglose hidráulica/eléctrica/aviónica/etc. —
     y propulsión).

Si SUAVE no está disponible o falla, `desglose_pesos()` devuelve None y
el llamador usa el desglose manual de respaldo.
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------
# Shims de compatibilidad (antes de cualquier import de SUAVE)
# --------------------------------------------------------------------------
def _aplicar_shims() -> None:
    import numpy as np

    try:
        import scipy.integrate as si

        if not hasattr(si, "cumtrapz") and hasattr(si, "cumulative_trapezoid"):
            si.cumtrapz = si.cumulative_trapezoid
    except Exception:
        pass

    try:
        import scipy.misc as sm

        if not hasattr(sm, "derivative"):

            def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
                # Reemplazo de scipy.misc.derivative (eliminado en scipy 1.12)
                if n == 1 and order >= 3:
                    return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (
                        2 * dx
                    )
                if n == 2 and order >= 3:
                    return (
                        func(x0 + dx, *args) - 2 * func(x0, *args)
                        + func(x0 - dx, *args)
                    ) / dx**2
                raise NotImplementedError(
                    "shim de derivative solo soporta n=1,2 order=3"
                )

            sm.derivative = derivative
    except Exception:
        pass

    for nombre, tipo in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("object", object),
        ("complex", complex),
    ):
        if nombre not in np.__dict__:
            setattr(np, nombre, tipo)
    if "NaN" not in np.__dict__:
        np.NaN = np.nan
    if "Inf" not in np.__dict__:
        np.Inf = np.inf


_aplicar_shims()


def importar_suave():
    """Devuelve (SUAVE, GA) o (None, error_str)."""
    try:
        import SUAVE
        from SUAVE.Methods.Weights.Correlations.General_Aviation import (
            empty as ga_empty,
        )

        return (SUAVE, ga_empty), None
    except Exception as exc:  # pragma: no cover
        return None, f"{type(exc).__name__}: {exc}"


def suave_disponible() -> bool:
    mod, _ = importar_suave()
    return mod is not None


# --------------------------------------------------------------------------
# Construcción del vehicle SUAVE y llamada al desglose
# --------------------------------------------------------------------------
def desglose_pesos(geo: dict) -> dict | None:
    """
    Ejecuta el desglose SUAVE para la geometría/pesos dados.

    Parametros
    ----------
    geo : dict con claves
        mtow, comb_total, s_ala, b, mac, c_root, lambda_wing, barrido_le,
        x_le_ala, x_ac, x_le_ht, x_le_vt, s_ht, b_ht, c_root_ht, lambda_ht,
        s_vt, b_vt, c_root_vt, lambda_vt, l_fus, d_fuselaje, mojada,
        masa_grupo, n_motores, n_pax, n_asientos

    Devuelve
    --------
    dict con masas por componente (kg) o None si SUAVE falla.
    """
    mod, _err = importar_suave()
    if mod is None:
        return None
    SUAVE, ga_empty = mod

    try:
        return _construir_y_correr(SUAVE, ga_empty, geo)
    except Exception as exc:
        # Deja constancia para que el llamador use el respaldo manual
        import sys

        print(f"[suave] falla ({type(exc).__name__}: {exc}); "
              f"se usa desglose manual.", file=sys.stderr)
        return None


def _construir_y_correr(SUAVE, ga_empty, geo: dict) -> dict:
    Data = SUAVE.Core.Data

    vehicle = Data()
    vehicle.reference_area = geo["s_ala"]

    # Cargas del envolvente (FAR 25 / clase transporte regional)
    vehicle.envelope = Data()
    vehicle.envelope.limit_load = 2.5
    vehicle.envelope.ultimate_load = 3.75

    vehicle.mass_properties = Data()
    vehicle.mass_properties.max_takeoff = geo["mtow"]
    vehicle.mass_properties.cargo = 0.0

    vehicle.passengers = geo["n_pax"]

    # Condiciones de diseño del crucero (FL200, 480 km/h TAS)
    vehicle.design_dynamic_pressure = 5801.0  # Pa
    vehicle.design_mach_number = 0.422

    # Red turboprop (rama "else" de GA.empty: masa definida por nosotros)
    vehicle.networks = Data()
    net = Data()
    net.number_of_engines = geo["n_motores"]
    net.mass_properties = Data()
    net.mass_properties.mass = geo["masa_grupo"]
    vehicle.networks.turboprop = net

    # Combustible (tanques de ala)
    vehicle.fuel = Data()
    vehicle.fuel.mass_properties = Data()
    vehicle.fuel.mass_properties.mass = geo["comb_total"]
    vehicle.fuel.density = 800.0  # kg/m³ Jet-A
    vehicle.fuel.number_of_tanks = 2
    vehicle.fuel.internal_volume = geo["comb_total"] / 800.0

    # --- Alas y colas -------------------------------------------------------
    lam_qc = math.radians(
        math.degrees(math.atan(math.tan(math.radians(geo["barrido_le"])))
                     - 0.5 * geo["c_root"] * (1 - geo["lambda_wing"])
                     / geo["b"])
    )
    lam_ht_qc = math.atan(
        -0.5 * geo["c_root_ht"] * (1 - geo["lambda_ht"]) / geo["b_ht"]
    )
    lam_vt_qc = math.radians(geo.get("barrido_vt_le", 35.0))

    vehicle.wings = Data()

    wing = Data()
    wing.spans = Data()
    wing.spans.projected = geo["b"]
    wing.taper = geo["lambda_wing"]
    wing.thickness_to_chord = 0.12
    wing.sweeps = Data()
    wing.sweeps.quarter_chord = lam_qc
    wing.origin = [[geo["x_le_ala"], 0.0, 1.38]]
    wing.aerodynamic_center = [geo["mac"] / 4.0, 0.0, 0.0]
    wing.mass_properties = Data()
    vehicle.wings.main_wing = wing

    c_ht = geo["c_root_ht"]
    ht = Data()
    ht.areas = Data()
    ht.areas.reference = geo["s_ht"]
    ht.spans = Data()
    ht.spans.projected = geo["b_ht"]
    ht.taper = geo["lambda_ht"]
    ht.thickness_to_chord = 0.12
    ht.sweeps = Data()
    ht.sweeps.quarter_chord = lam_ht_qc
    ht.origin = [[geo["x_le_ht"], 0.0, 0.10]]
    ht.aerodynamic_center = [c_ht / 4.0, 0.0, 0.0]
    ht.mass_properties = Data()
    vehicle.wings.horizontal_stabilizer = ht

    vt = Data()
    vt.areas = Data()
    vt.areas.reference = geo["s_vt"]
    vt.spans = Data()
    vt.spans.projected = geo["b_vt"]
    vt.taper = geo["lambda_vt"]
    vt.thickness_to_chord = 0.12
    vt.sweeps = Data()
    vt.sweeps.quarter_chord = lam_vt_qc
    vt.t_tail = "no"
    vt.origin = [[geo["x_le_vt"], 0.0, 0.0]]
    vt.aerodynamic_center = [geo["c_root_vt"] / 4.0, 0.0, 0.0]
    vt.mass_properties = Data()
    vehicle.wings.vertical_stabilizer = vt

    # --- Fuselaje -----------------------------------------------------------
    d = geo["d_fuselaje"]
    fus = Data()
    fus.areas = Data()
    fus.areas.wetted = geo["mojada"]
    fus.differential_pressure = 51000.0  # Pa (~0,5 bar, cabina presurizada)
    fus.width = d
    fus.heights = Data()
    fus.heights.maximum = d
    fus.lengths = Data()
    fus.lengths.structure = geo["l_fus"]
    fus.mass_properties = Data()
    fus.mass_properties.volume = math.pi / 4 * d * d * geo["l_fus"] * 0.85
    fus.number_coach_seats = geo["n_asientos"]
    vehicle.fuselages = Data()
    vehicle.fuselages.fuselage = fus

    # --- Tren de aterrizaje -------------------------------------------------
    vehicle.landing_gear = Data()
    vehicle.landing_gear.main = Data()
    vehicle.landing_gear.main.strut_length = 0.95
    vehicle.landing_gear.main.mass_properties = Data()
    vehicle.landing_gear.nose = Data()
    vehicle.landing_gear.nose.strut_length = 0.60
    vehicle.landing_gear.nose.mass_properties = Data()

    # --- Aviónica y climatización -------------------------------------------
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.mass_properties.uninstalled = 250.0  # kg (suite regional)
    vehicle.avionics = avionics
    vehicle.air_conditioner = Data()  # marca presencia de ECS/press.
    vehicle.air_conditioner.mass_properties = Data()
    vehicle.payload = Data()  # lo llena el packup de empty()

    out = ga_empty(vehicle)

    return {
        "estructura_fuselaje": out.structures.fuselage,
        "estructura_ala": out.structures.wing,
        "estructura_ht": out.structures.horizontal_tail,
        "estructura_vt": out.structures.vertical_tail,
        "tren_principal": out.structures.main_landing_gear,
        "tren_nariz": out.structures.nose_landing_gear,
        "propulsion": out.propulsion_breakdown.total,
        "sis_control": out.systems_breakdown.control_systems,
        "sis_hidraulica": out.systems_breakdown.hydraulics,
        "sis_avionica": out.systems_breakdown.avionics,
        "sis_electrico": out.systems_breakdown.electrical,
        "sis_clima": out.systems_breakdown.air_conditioner,
        "sis_amueblado": out.systems_breakdown.furnish,
        "sis_combustible": out.propulsion_breakdown.fuel_system,
        "_suave_empty": out.empty,  # estructura+propulsión+sistemas (SUAVE)
    }


if __name__ == "__main__":
    # Prueba rápida de importación
    mod, error = importar_suave()
    if mod is None:
        print("SUAVE NO disponible:", error)
    else:
        print("SUAVE disponible:",
              getattr(mod[0], "__version__", "?"))
