import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Gaussian NMR Boltzmann Averaging App", layout="wide")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1

st.title("Gaussian NMR Boltzmann Averaging App")
st.caption("Ver. 1.3")
st.write(
    "Upload Gaussian opt+freq logs and GIAO logs, match conformers by filename, "
    "extract SCF or Gibbs free energies, calculate Boltzmann-averaged isotropic shieldings, "
    "and convert them to chemical shifts using manual references, a TMS log, or linear scaling."
)

# =========================================================
# Session state initialization
# =========================================================
if "equivalent_groups_ui" not in st.session_state:
    st.session_state["equivalent_groups_ui"] = []

if "latest_atom_table" not in st.session_state:
    st.session_state["latest_atom_table"] = pd.DataFrame(columns=["atom_index", "element"])

# =========================================================
# Helper functions
# =========================================================
def extract_conf_id(filename: str):
    """
    Extract a conformer ID from filename.
    Priority:
      1) trailing integer before .log/.out
      2) confXX pattern
      3) fallback to filename stem
    """
    m = re.search(r"(\d+)\.(log|out)$", filename, re.IGNORECASE)
    if m:
        return m.group(1)

    m2 = re.search(r"conf[_\- ]*(\d+)", filename, re.IGNORECASE)
    if m2:
        return m2.group(1)

    stem = re.sub(r"\.(log|out)$", "", filename, flags=re.IGNORECASE)
    return stem


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
            except Exception:
                pass

    key2 = "Sum of electronic and thermal Free Energies"
    for line in text.splitlines():
        if key2 in line:
            try:
                return float(line.split()[-1])
            except Exception:
                pass

    return None


def extract_last_scf_energy(text: str):
    pattern = re.compile(r"SCF Done:\s+E\([RU]?[A-Za-z0-9]+\)\s*=\s*(-?\d+\.\d+)")
    matches = pattern.findall(text)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
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
        rows.append(
            {
                "atom_index": int(m.group(1)),
                "element": m.group(2),
                "shielding": float(m.group(3)),
            }
        )
    return pd.DataFrame(rows)


def get_tms_reference_from_log(text):
    df = extract_isotropic_shieldings(text)

    if df.empty:
        return None, None, "No isotropic shielding entries were found in the TMS log."

    h_df = df[df["element"] == "H"].copy()
    c_df = df[df["element"] == "C"].copy()

    if h_df.empty:
        return None, None, "No hydrogen shielding values were found in the TMS log."

    if c_df.empty:
        return None, None, "No carbon shielding values were found in the TMS log."

    ref_H = h_df["shielding"].mean()
    ref_C = c_df["shielding"].mean()

    return ref_H, ref_C, None


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

    for cid, w in zip(conf_ids, weights):
        col = f"shielding_{cid}"
        weighted_col = f"weighted_{cid}"
        out[weighted_col] = out[col] * w

    weighted_cols = [f"weighted_{cid}" for cid in conf_ids]
    out["shielding_boltzmann"] = out[weighted_cols].sum(axis=1)
    return out


def shielding_to_shift(
    df,
    mode="manual_reference",
    ref_H=31.5,
    ref_C=185.0,
    slope_H=1.0,
    intercept_H=31.5,
    slope_C=1.0,
    intercept_C=185.0,
):
    out = df.copy()
    shifts = []

    for _, row in out.iterrows():
        s = row["shielding_boltzmann"]
        el = row["element"]

        if mode in ["manual_reference", "tms_log"]:
            if el == "H":
                delta = ref_H - s
            elif el == "C":
                delta = ref_C - s
            else:
                delta = np.nan
        elif mode == "linear":
            if el == "H":
                delta = intercept_H - slope_H * s
            elif el == "C":
                delta = intercept_C - slope_C * s
            else:
                delta = np.nan
        else:
            delta = np.nan

        shifts.append(delta)

    out["chemical_shift"] = shifts
    return out


def average_equivalent_atoms(df, groups):
    """
    For each user-defined equivalent atom group, average:
      - each conformer's shielding
      - weighted conformer columns
      - Boltzmann averaged shielding
      - chemical shift if available
    """
    results = []

    value_cols = [
        c
        for c in df.columns
        if c.startswith("shielding_") or c.startswith("weighted_")
    ]
    if "shielding_boltzmann" in df.columns:
        value_cols.append("shielding_boltzmann")
    if "chemical_shift" in df.columns:
        value_cols.append("chemical_shift")

    value_cols = list(dict.fromkeys(value_cols))

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
            "element": element_label,
        }

        for col in value_cols:
            row[col] = sub[col].mean()

        results.append(row)

    if results:
        return pd.DataFrame(results)

    return pd.DataFrame(columns=["group_label", "atom_indices", "n_atoms", "element"])


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def get_registered_atom_set(groups):
    atom_set = set()
    for g in groups:
        atom_set.update(g["atoms"])
    return atom_set


def make_atom_option_labels(atom_df):
    labels = []
    for _, row in atom_df.iterrows():
        labels.append(f'{row["atom_index"]} ({row["element"]})')
    return labels


def parse_atom_indices_from_labels(labels):
    atoms = []
    for label in labels:
        m = re.match(r"^\s*(\d+)", str(label))
        if m:
            atoms.append(int(m.group(1)))
    return sorted(set(atoms))


# =========================================================
# Sidebar settings
# =========================================================
st.sidebar.header("Settings")

temperature = st.sidebar.number_input("Temperature (K)", value=298.15, step=1.0)

energy_mode = st.sidebar.radio(
    "Energy to use for Boltzmann weighting",
    ["Gibbs free energy", "SCF energy"],
    index=0,
)

shift_mode = st.sidebar.radio(
    "Chemical shift conversion method",
    ["Manual reference shielding", "TMS log file", "Linear scaling"],
    index=0,
)

ref_H = None
ref_C = None
slope_H = None
intercept_H = None
slope_C = None
intercept_C = None

if shift_mode == "Manual reference shielding":
    ref_H = st.sidebar.number_input("Reference shielding for 1H", value=31.5)
    ref_C = st.sidebar.number_input("Reference shielding for 13C", value=185.0)

elif shift_mode == "TMS log file":
    tms_file = st.sidebar.file_uploader(
        "Upload TMS GIAO log file",
        type=["log", "out"],
        accept_multiple_files=False,
        key="tms_log",
    )

    if tms_file is not None:
        tms_text = read_text(tms_file)
        ref_H, ref_C, tms_error = get_tms_reference_from_log(tms_text)

        if tms_error:
            st.sidebar.error(tms_error)
        else:
            st.sidebar.success("TMS reference extracted successfully.")
            st.sidebar.write(f"TMS 1H reference shielding: {ref_H:.4f}")
            st.sidebar.write(f"TMS 13C reference shielding: {ref_C:.4f}")
    else:
        st.sidebar.info("Please upload a TMS GIAO log file.")

elif shift_mode == "Linear scaling":
    slope_H = st.sidebar.number_input("Slope for 1H", value=1.0)
    intercept_H = st.sidebar.number_input("Intercept for 1H", value=31.5)
    slope_C = st.sidebar.number_input("Slope for 13C", value=1.0)
    intercept_C = st.sidebar.number_input("Intercept for 13C", value=185.0)

element_filter = st.sidebar.selectbox(
    "Element filter for display",
    ["All", "H", "C", "Other"],
    index=0,
)

# =========================================================
# Upload files
# =========================================================
st.subheader("1. Upload files")

opt_files = st.file_uploader(
    "Upload opt+freq log files",
    type=["log", "out"],
    accept_multiple_files=True,
    key="opt_files",
)

giao_files = st.file_uploader(
    "Upload GIAO log files",
    type=["log", "out"],
    accept_multiple_files=True,
    key="giao_files",
)

result_df = None
per_conf_df = None
valid_df = None

if opt_files and giao_files:
    # -----------------------------------------------------
    # Parse opt+freq logs
    # -----------------------------------------------------
    opt_records = []
    for f in opt_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        gibbs = extract_gibbs_free_energy(text)
        scf = extract_last_scf_energy(text)
        normal = check_normal_termination(text)

        opt_records.append(
            {
                "conf_id": cid,
                "opt_filename": f.name,
                "gibbs_hartree": gibbs,
                "scf_hartree": scf,
                "opt_normal_termination": normal,
            }
        )

    opt_df = pd.DataFrame(opt_records)

    # -----------------------------------------------------
    # Parse GIAO logs
    # -----------------------------------------------------
    giao_records = []
    shielding_map = {}

    for f in giao_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        normal = check_normal_termination(text)
        shielding_df = extract_isotropic_shieldings(text)

        giao_records.append(
            {
                "conf_id": cid,
                "giao_filename": f.name,
                "n_atoms_found": len(shielding_df),
                "giao_normal_termination": normal,
            }
        )

        shielding_map[cid] = shielding_df

    giao_df = pd.DataFrame(giao_records)

    # -----------------------------------------------------
    # Match logs
    # -----------------------------------------------------
    pair_df = pd.merge(opt_df, giao_df, on="conf_id", how="inner")

    st.subheader("2. Matched conformers")
    st.dataframe(pair_df, use_container_width=True)

    if energy_mode == "Gibbs free energy":
        energy_col = "gibbs_hartree"
    else:
        energy_col = "scf_hartree"

    valid_df = pair_df[
        pair_df["conf_id"].notna()
        & pair_df[energy_col].notna()
        & pair_df["opt_normal_termination"]
        & pair_df["giao_normal_termination"]
        & (pair_df["n_atoms_found"] > 0)
    ].copy()

    if len(valid_df) == 0:
        st.error("No valid matched conformers were found.")
        st.stop()

    if shift_mode == "TMS log file" and (ref_H is None or ref_C is None):
        st.error("Please upload a valid TMS GIAO log file before calculating chemical shifts.")
        st.stop()

    # -----------------------------------------------------
    # Boltzmann weights
    # -----------------------------------------------------
    rel_kcal, weights = boltzmann_weights(valid_df[energy_col].values, temperature=temperature)
    valid_df["energy_used_hartree"] = valid_df[energy_col]
    valid_df["relative_energy_kcal"] = rel_kcal
    valid_df["boltzmann_weight"] = weights

    st.subheader("3. Energies and Boltzmann weights")
    st.dataframe(valid_df, use_container_width=True)

    # -----------------------------------------------------
    # Per-conformer shielding table
    # -----------------------------------------------------
    conf_ids = valid_df["conf_id"].tolist()
    per_conf_df = build_per_conformer_shielding_table(shielding_map, conf_ids)

    # store atom table for UI
    atom_table_full = per_conf_df[["atom_index", "element"]].drop_duplicates().sort_values("atom_index").reset_index(drop=True)
    st.session_state["latest_atom_table"] = atom_table_full.copy()

    if element_filter == "H":
        per_conf_df = per_conf_df[per_conf_df["element"] == "H"].copy()
    elif element_filter == "C":
        per_conf_df = per_conf_df[per_conf_df["element"] == "C"].copy()
    elif element_filter == "Other":
        per_conf_df = per_conf_df[~per_conf_df["element"].isin(["H", "C"])].copy()

    st.subheader("4. Isotropic shielding table for each conformer")
    st.dataframe(per_conf_df, use_container_width=True)

    # -----------------------------------------------------
    # Boltzmann averaged table
    # -----------------------------------------------------
    avg_df = add_boltzmann_average(per_conf_df, conf_ids, weights)

    if shift_mode == "Manual reference shielding":
        result_df = shielding_to_shift(
            avg_df,
            mode="manual_reference",
            ref_H=ref_H,
            ref_C=ref_C,
        )
    elif shift_mode == "TMS log file":
        result_df = shielding_to_shift(
            avg_df,
            mode="tms_log",
            ref_H=ref_H,
            ref_C=ref_C,
        )
    else:
        result_df = shielding_to_shift(
            avg_df,
            mode="linear",
            slope_H=slope_H,
            intercept_H=intercept_H,
            slope_C=slope_C,
            intercept_C=intercept_C,
        )

    st.subheader("5. Per-atom Boltzmann-averaged shielding / shift table")
    st.dataframe(result_df, use_container_width=True)

# =========================================================
# Equivalent atom group UI
# =========================================================
st.subheader("6. Equivalent atom groups")

atom_df_ui = st.session_state["latest_atom_table"].copy()

if atom_df_ui.empty:
    st.info("After valid GIAO files are uploaded, the atom list will appear here.")
else:
    ui_filter = st.selectbox(
        "Filter atoms for group selection",
        ["All", "H", "C", "Other"],
        index=0,
        key="eq_ui_filter",
    )

    if ui_filter == "H":
        atom_df_ui = atom_df_ui[atom_df_ui["element"] == "H"].copy()
    elif ui_filter == "C":
        atom_df_ui = atom_df_ui[atom_df_ui["element"] == "C"].copy()
    elif ui_filter == "Other":
        atom_df_ui = atom_df_ui[~atom_df_ui["element"].isin(["H", "C"])].copy()

    registered_atoms = get_registered_atom_set(st.session_state["equivalent_groups_ui"])
    atom_df_ui["already_registered"] = atom_df_ui["atom_index"].isin(registered_atoms)

    st.write("Available atoms")
    st.dataframe(atom_df_ui, use_container_width=True)

    group_label = st.text_input("Group label", value="", key="eq_group_label")

    atom_option_labels = make_atom_option_labels(atom_df_ui[["atom_index", "element"]])
    selected_atom_labels = st.multiselect(
        "Select equivalent atom indices",
        options=atom_option_labels,
        default=[],
        key="eq_selected_atoms",
    )

    selected_atoms = parse_atom_indices_from_labels(selected_atom_labels)

    if selected_atoms:
        overlapping_atoms = sorted(set(selected_atoms) & registered_atoms)
        if overlapping_atoms:
            st.warning(
                "These atom indices are already included in existing groups: "
                + ", ".join(map(str, overlapping_atoms))
            )

    if st.button("Add equivalent atom group", key="add_eq_group"):
        clean_label = group_label.strip()

        if not clean_label:
            st.warning("Please enter a group label.")
        elif not selected_atoms:
            st.warning("Please select at least one atom.")
        else:
            st.session_state["equivalent_groups_ui"].append(
                {
                    "label": clean_label,
                    "atoms": selected_atoms,
                }
            )
            st.rerun()

# =========================================================
# Registered groups
# =========================================================
if st.session_state["equivalent_groups_ui"]:
    st.write("Registered equivalent atom groups")

    registered_rows = []
    for g in st.session_state["equivalent_groups_ui"]:
        registered_rows.append(
            {
                "label": g["label"],
                "atom_indices": ", ".join(map(str, g["atoms"])),
                "n_atoms": len(g["atoms"]),
            }
        )
    st.dataframe(pd.DataFrame(registered_rows), use_container_width=True)

    for i, g in enumerate(st.session_state["equivalent_groups_ui"]):
        col1, col2, col3 = st.columns([2, 6, 1])
        col1.write(f'**{g["label"]}**')
        col2.write(", ".join(map(str, g["atoms"])))
        if col3.button("Delete", key=f"delete_group_{i}"):
            st.session_state["equivalent_groups_ui"].pop(i)
            st.rerun()

# =========================================================
# Equivalent-atom averaged result + downloads
# =========================================================
if result_df is not None:
    if st.session_state["equivalent_groups_ui"]:
        eq_df = average_equivalent_atoms(result_df, st.session_state["equivalent_groups_ui"])
        st.subheader("7. Equivalent-atom averaged table")
        st.dataframe(eq_df, use_container_width=True)

    st.subheader("8. Download outputs")

    if per_conf_df is not None:
        st.download_button(
            label="Download per-conformer shielding table (CSV)",
            data=dataframe_to_csv_bytes(per_conf_df),
            file_name="per_conformer_isotropic_shieldings.csv",
            mime="text/csv",
        )

    st.download_button(
        label="Download per-atom Boltzmann averaged table (CSV)",
        data=dataframe_to_csv_bytes(result_df),
        file_name="boltzmann_averaged_nmr.csv",
        mime="text/csv",
    )

    if valid_df is not None:
        st.download_button(
            label="Download energy / weight table (CSV)",
            data=dataframe_to_csv_bytes(valid_df),
            file_name="boltzmann_weights.csv",
            mime="text/csv",
        )

    if st.session_state["equivalent_groups_ui"]:
        eq_df = average_equivalent_atoms(result_df, st.session_state["equivalent_groups_ui"])
        st.download_button(
            label="Download equivalent-atom averaged table (CSV)",
            data=dataframe_to_csv_bytes(eq_df),
            file_name="equivalent_atom_averaged_nmr.csv",
            mime="text/csv",
        )
