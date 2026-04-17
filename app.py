import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from math import exp

st.set_page_config(page_title="Gaussian NMR Boltzmann Averaging App", layout="wide")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1

st.title("Gaussian NMR Boltzmann Averaging App")
st.caption("Ver. 1.0")
st.write("Upload Gaussian opt+freq logs and GIAO logs, match conformers by filename, and calculate Boltzmann-averaged NMR shieldings / shifts.")

# -----------------------------
# helper functions
# -----------------------------
def extract_conf_id(filename: str):
    m = re.search(r"(\d+)\.log$", filename)
    return m.group(1) if m else None

def read_text(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")

def check_normal_termination(text: str):
    return "Normal termination of Gaussian" in text

def extract_gibbs_free_energy(text: str):
    key = "Sum of electronic and thermal Free Energies="
    for line in text.splitlines():
        if key in line:
            try:
                return float(line.split("=")[-1].strip())
            except:
                pass

    key2 = "Sum of electronic and thermal Free Energies"
    for line in text.splitlines():
        if key2 in line:
            try:
                return float(line.split()[-1])
            except:
                pass
    return None

def extract_isotropic_shieldings(text: str):
    """
    Gaussian GIAO log lines like:
      69  C    Isotropic =   142.0260   Anisotropy =    79.9308
      70  H    Isotropic =    28.4701   Anisotropy =     9.7612
    """
    pattern = re.compile(
        r"^\s*(\d+)\s+([A-Z][a-z]?)\s+Isotropic\s*=\s*(-?\d+\.\d+)",
        re.MULTILINE
    )
    rows = []
    for m in pattern.finditer(text):
        atom_index = int(m.group(1))
        element = m.group(2)
        shielding = float(m.group(3))
        rows.append({
            "atom_index": atom_index,
            "element": element,
            "shielding": shielding
        })
    return pd.DataFrame(rows)

def boltzmann_weights(energies_hartree, temperature=298.15):
    energies_hartree = np.array(energies_hartree, dtype=float)
    rel_kcal = (energies_hartree - energies_hartree.min()) * HARTREE_TO_KCAL
    weights = np.exp(-rel_kcal / (R_KCAL * temperature))
    weights /= weights.sum()
    return rel_kcal, weights

def average_shieldings(shielding_tables, weights, conf_ids):
    merged = None

    for df, w, cid in zip(shielding_tables, weights, conf_ids):
        temp = df.copy()
        temp[f"shielding_{cid}"] = temp["shielding"]
        temp[f"weighted_{cid}"] = temp["shielding"] * w
        temp = temp[["atom_index", "element", f"shielding_{cid}", f"weighted_{cid}"]]

        if merged is None:
            merged = temp
        else:
            merged = pd.merge(merged, temp, on=["atom_index", "element"], how="outer")

    weighted_cols = [c for c in merged.columns if c.startswith("weighted_")]
    merged["shielding_boltzmann"] = merged[weighted_cols].sum(axis=1)
    return merged

def shielding_to_shift(df, mode="linear", ref_H=31.5, ref_C=185.0,
                       slope_H=1.0, intercept_H=31.5,
                       slope_C=1.0, intercept_C=185.0):
    out = df.copy()

    shifts = []
    for _, row in out.iterrows():
        s = row["shielding_boltzmann"]
        el = row["element"]

        if mode == "reference":
            if el == "H":
                delta = ref_H - s
            elif el == "C":
                delta = ref_C - s
            else:
                delta = np.nan
        else:
            if el == "H":
                delta = intercept_H - slope_H * s
            elif el == "C":
                delta = intercept_C - slope_C * s
            else:
                delta = np.nan

        shifts.append(delta)

    out["chemical_shift"] = shifts
    return out

# -----------------------------
# sidebar
# -----------------------------
st.sidebar.header("Settings")

temperature = st.sidebar.number_input("Temperature (K)", value=298.15, step=1.0)

shift_mode = st.sidebar.radio(
    "Chemical shift conversion",
    ["reference", "linear"],
    index=0
)

if shift_mode == "reference":
    ref_H = st.sidebar.number_input("Reference shielding for 1H", value=31.5)
    ref_C = st.sidebar.number_input("Reference shielding for 13C", value=185.0)
else:
    slope_H = st.sidebar.number_input("Slope for 1H", value=1.0)
    intercept_H = st.sidebar.number_input("Intercept for 1H", value=31.5)
    slope_C = st.sidebar.number_input("Slope for 13C", value=1.0)
    intercept_C = st.sidebar.number_input("Intercept for 13C", value=185.0)

# -----------------------------
# upload
# -----------------------------
st.subheader("1. Upload files")

opt_files = st.file_uploader(
    "Upload opt+freq log files",
    type=["log", "out"],
    accept_multiple_files=True,
    key="opt"
)

giao_files = st.file_uploader(
    "Upload GIAO log files",
    type=["log", "out"],
    accept_multiple_files=True,
    key="giao"
)

if opt_files and giao_files:
    # parse optfreq
    opt_records = []
    for f in opt_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        gibbs = extract_gibbs_free_energy(text)
        normal = check_normal_termination(text)

        opt_records.append({
            "conf_id": cid,
            "opt_filename": f.name,
            "gibbs_hartree": gibbs,
            "opt_normal_termination": normal,
            "opt_text": text
        })

    opt_df = pd.DataFrame(opt_records)

    # parse giao
    giao_records = []
    shielding_map = {}

    for f in giao_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        normal = check_normal_termination(text)
        shielding_df = extract_isotropic_shieldings(text)

        giao_records.append({
            "conf_id": cid,
            "giao_filename": f.name,
            "n_atoms_found": len(shielding_df),
            "giao_normal_termination": normal,
        })
        shielding_map[cid] = shielding_df

    giao_df = pd.DataFrame(giao_records)

    # match
    pair_df = pd.merge(opt_df.drop(columns=["opt_text"]), giao_df, on="conf_id", how="inner")

    st.subheader("2. Matched conformers")
    st.dataframe(pair_df, use_container_width=True)

    valid_df = pair_df[
        pair_df["conf_id"].notna() &
        pair_df["gibbs_hartree"].notna() &
        pair_df["opt_normal_termination"] &
        pair_df["giao_normal_termination"] &
        (pair_df["n_atoms_found"] > 0)
    ].copy()

    if len(valid_df) == 0:
        st.error("No valid matched conformers were found.")
    else:
        rel_kcal, weights = boltzmann_weights(valid_df["gibbs_hartree"].values, temperature=temperature)
        valid_df["relative_energy_kcal"] = rel_kcal
        valid_df["boltzmann_weight"] = weights

        st.subheader("3. Energies and Boltzmann weights")
        st.dataframe(valid_df, use_container_width=True)

        conf_ids = valid_df["conf_id"].tolist()
        shielding_tables = [shielding_map[cid] for cid in conf_ids]

        avg_df = average_shieldings(shielding_tables, weights, conf_ids)

        if shift_mode == "reference":
            result_df = shielding_to_shift(
                avg_df,
                mode="reference",
                ref_H=ref_H,
                ref_C=ref_C
            )
        else:
            result_df = shielding_to_shift(
                avg_df,
                mode="linear",
                slope_H=slope_H,
                intercept_H=intercept_H,
                slope_C=slope_C,
                intercept_C=intercept_C
            )

        st.subheader("4. Boltzmann-averaged shielding / shift table")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download averaged NMR table (CSV)",
            data=csv_bytes,
            file_name="boltzmann_averaged_nmr.csv",
            mime="text/csv"
        )

        weight_csv = valid_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download energy / weight table (CSV)",
            data=weight_csv,
            file_name="boltzmann_weights.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload both opt+freq logs and GIAO logs.")
