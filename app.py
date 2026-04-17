import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Gaussian NMR Boltzmann Averaging App", layout="wide")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1

st.title("Gaussian NMR Boltzmann Averaging App")
st.caption("Ver. 1.1")
st.write(
    "Upload Gaussian opt+freq logs and GIAO logs, match conformers by filename, "
    "extract SCF or Gibbs free energies, calculate Boltzmann-averaged isotropic shieldings, "
    "and optionally average equivalent atoms."
)

# =========================
# helper functions
# =========================
def extract_conf_id(filename: str):
    """
    Extract trailing integer before .log / .out
    Examples:
      conf01_optfreq.log -> None with this simple rule
      ..._3.log -> 3
    """
    m = re.search(r"(\d+)\.(log|out)$", filename, re.IGNORECASE)
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


def extract_last_scf_energy(text: str):
    """
    Extract the last SCF Done energy from Gaussian log.
    Example:
      SCF Done:  E(RB3LYP) =  -1234.56789012     A.U. after ...
    """
    pattern = re.compile(r"SCF Done:\s+E\([RU]?[A-Z0-9]+\)\s+=\s+(-?\d+\.\d+)")
    matches = pattern.findall(text)
    if matches:
        try:
            return float(matches[-1])
        except:
            return None
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
        rows.append({
            "atom_index": int(m.group(1)),
            "element": m.group(2),
            "shielding": float(m.group(3))
        })
    return pd.DataFrame(rows)


def boltzmann_weights(energies_hartree, temperature=298.15):
    energies_hartree = np.array(energies_hartree, dtype=float)
    rel_kcal = (energies_hartree - energies_hartree.min()) * HARTREE_TO_KCAL
    weights = np.exp(-rel_kcal / (R_KCAL * temperature))
    weights /= weights.sum()
    return rel_kcal, weights


def build_per_conformer_shielding_table(shielding_map, conf_ids):
    """
    Merge shielding tables from each conformer into one table:
      atom_index | element | shielding_conf1 | shielding_conf2 | ...
    """
    merged = None
    for cid in conf_ids:
        df = shielding_map[cid].copy()
        df = df.rename(columns={"shielding": f"shielding_{cid}"})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["atom_index", "element"], how="outer")
    return merged


def add_boltzmann_average(per_conf_df, conf_ids, weights):
    out = per_conf_df.copy()
    weighted_sum = np.zeros(len(out), dtype=float)

    for cid, w in zip(conf_ids, weights):
        col = f"shielding_{cid}"
        weighted_col = f"weighted_{cid}"
        out[weighted_col] = out[col] * w
        weighted_sum += out[col].astype(float).values * w

    out["shielding_boltzmann"] = weighted_sum
    return out


def shielding_to_shift(
    df,
    mode="reference",
    ref_H=31.5,
    ref_C=185.0,
    slope_H=1.0,
    intercept_H=31.5,
    slope_C=1.0,
    intercept_C=185.0
):
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


def parse_equivalent_groups(text):
    """
    Input format example:
      H_a: 1,2,3
      OMe: 45,46,47
      C_ring: 10,12

    Returns:
      [
        {"label": "H_a", "atoms": [1,2,3]},
        {"label": "OMe", "atoms": [45,46,47]},
      ]
    """
    groups = []
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    for i, line in enumerate(lines, start=1):
        if ":" in line:
            label, atom_part = line.split(":", 1)
            label = label.strip()
            atom_part = atom_part.strip()
        else:
            label = f"group_{i}"
            atom_part = line

        atoms = []
        for token in atom_part.split(","):
            token = token.strip()
            if token:
                try:
                    atoms.append(int(token))
                except:
                    pass

        atoms = sorted(set(atoms))
        if atoms:
            groups.append({"label": label, "atoms": atoms})

    return groups


def average_equivalent_atoms(df, groups, conf_ids):
    """
    For each user-defined equivalent atom group, average:
      - each conformer's shielding
      - weighted conformer columns
      - Boltzmann averaged shielding
      - chemical shift if available
    """
    results = []

    numeric_cols = [c for c in df.columns if c.startswith("shielding_") or c.startswith("weighted_")]
    has_shift = "chemical_shift" in df.columns

    for group in groups:
        atoms = group["atoms"]
        sub = df[df["atom_index"].isin(atoms)].copy()

        if sub.empty:
            continue

        elements = sorted(sub["element"].dropna().unique().tolist())
        element_label = "/".join(elements) if elements else ""

        row = {
            "group_label": group["label"],
            "atom_indices": ",".join(map(str, atoms)),
            "n_atoms": len(atoms),
            "element": element_label
        }

        for col in numeric_cols:
            row[col] = sub[col].mean()

        if "shielding_boltzmann" in sub.columns:
            row["shielding_boltzmann"] = sub["shielding_boltzmann"].mean()

        if has_shift:
            row["chemical_shift"] = sub["chemical_shift"].mean()

        results.append(row)

    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=["group_label", "atom_indices", "n_atoms", "element"])


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# =========================
# sidebar
# =========================
st.sidebar.header("Settings")

temperature = st.sidebar.number_input("Temperature (K)", value=298.15, step=1.0)

energy_mode = st.sidebar.radio(
    "Energy to use for Boltzmann weighting",
    ["Gibbs free energy", "SCF energy"],
    index=0
)

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

element_filter = st.sidebar.selectbox(
    "Element filter for display",
    ["All", "H", "C", "Other"],
    index=0
)

# =========================
# equivalent atoms input
# =========================
st.subheader("1. Optional equivalent atom groups")
eq_text = st.text_area(
    "Define equivalent atom groups (one group per line)",
    value="",
    height=140,
    placeholder="Examples:\nH_a: 1,2,3\nOMe: 45,46,47\nC_eq: 10,12"
)

equivalent_groups = parse_equivalent_groups(eq_text)

if equivalent_groups:
    st.write("Parsed equivalent groups:")
    st.dataframe(pd.DataFrame(equivalent_groups), use_container_width=True)

# =========================
# file upload
# =========================
st.subheader("2. Upload files")

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
    # -------------------------
    # parse optfreq files
    # -------------------------
    opt_records = []
    for f in opt_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        gibbs = extract_gibbs_free_energy(text)
        scf = extract_last_scf_energy(text)
        normal = check_normal_termination(text)

        opt_records.append({
            "conf_id": cid,
            "opt_filename": f.name,
            "gibbs_hartree": gibbs,
            "scf_hartree": scf,
            "opt_normal_termination": normal
        })

    opt_df = pd.DataFrame(opt_records)

    # -------------------------
    # parse GIAO files
    # -------------------------
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
            "giao_normal_termination": normal
        })

        shielding_map[cid] = shielding_df

    giao_df = pd.DataFrame(giao_records)

    # -------------------------
    # match files
    # -------------------------
    pair_df = pd.merge(opt_df, giao_df, on="conf_id", how="inner")

    st.subheader("3. Matched conformers")
    st.dataframe(pair_df, use_container_width=True)

    # -------------------------
    # choose energy column
    # -------------------------
    if energy_mode == "Gibbs free energy":
        energy_col = "gibbs_hartree"
    else:
        energy_col = "scf_hartree"

    valid_df = pair_df[
        pair_df["conf_id"].notna() &
        pair_df[energy_col].notna() &
        pair_df["opt_normal_termination"] &
        pair_df["giao_normal_termination"] &
        (pair_df["n_atoms_found"] > 0)
    ].copy()

    if len(valid_df) == 0:
        st.error("No valid matched conformers were found.")
    else:
        # -------------------------
        # Boltzmann weights
        # -------------------------
        rel_kcal, weights = boltzmann_weights(valid_df[energy_col].values, temperature=temperature)
        valid_df["energy_used_hartree"] = valid_df[energy_col]
        valid_df["relative_energy_kcal"] = rel_kcal
        valid_df["boltzmann_weight"] = weights

        st.subheader("4. Energies and Boltzmann weights")
        st.dataframe(valid_df, use_container_width=True)

        # -------------------------
        # per-conformer shielding table
        # -------------------------
        conf_ids = valid_df["conf_id"].tolist()
        per_conf_df = build_per_conformer_shielding_table(shielding_map, conf_ids)

        # Filter by element
        if element_filter == "H":
            per_conf_df = per_conf_df[per_conf_df["element"] == "H"].copy()
        elif element_filter == "C":
            per_conf_df = per_conf_df[per_conf_df["element"] == "C"].copy()
        elif element_filter == "Other":
            per_conf_df = per_conf_df[~per_conf_df["element"].isin(["H", "C"])].copy()

        st.subheader("5. Isotropic shielding table for each conformer")
        st.dataframe(per_conf_df, use_container_width=True)

        # -------------------------
        # Boltzmann averaged table
        # -------------------------
        avg_df = add_boltzmann_average(per_conf_df, conf_ids, weights)

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

        st.subheader("6. Per-atom Boltzmann-averaged shielding / shift table")
        st.dataframe(result_df, use_container_width=True)

        # -------------------------
        # Equivalent atom averages
        # -------------------------
        if equivalent_groups:
            eq_df = average_equivalent_atoms(result_df, equivalent_groups, conf_ids)
            st.subheader("7. Equivalent-atom averaged table")
            st.dataframe(eq_df, use_container_width=True)

            st.download_button(
                label="Download equivalent-atom averaged table (CSV)",
                data=dataframe_to_csv_bytes(eq_df),
                file_name="equivalent_atom_averaged_nmr.csv",
                mime="text/csv"
            )

        # -------------------------
        # downloads
        # -------------------------
        st.subheader("8. Download outputs")

        st.download_button(
            label="Download per-conformer shielding table (CSV)",
            data=dataframe_to_csv_bytes(per_conf_df),
            file_name="per_conformer_isotropic_shieldings.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download per-atom Boltzmann averaged table (CSV)",
            data=dataframe_to_csv_bytes(result_df),
            file_name="boltzmann_averaged_nmr.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download energy / weight table (CSV)",
            data=dataframe_to_csv_bytes(valid_df),
            file_name="boltzmann_weights.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload both opt+freq logs and GIAO logs.")
