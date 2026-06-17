import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path

st.set_page_config(page_title="Gaussian NMR Boltzmann Averaging App", layout="wide")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1
BOHR_TO_ANG = 0.529177210903

APP_VERSION = "2.1"

DEVELOPER_INFO = {
    "name": "Ken-ichi Nakashima",
    "affiliation_ja": "愛知学院大学 薬学部 薬用資源学講座",
    "affiliation_en": "Aichi-Gakuin University, School of Pharmacy, Laboratory of Natural Resources",
}

ATOM_PICKER = components.declare_component(
    "atom_picker_component",
    path="atom_picker_component",
)

ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 2: "He",
    3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo",
    43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce",
    59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy",
    67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
    83: "Bi", 84: "Po", 85: "At", 86: "Rn",
}

UI_TEXT = {
    "en": {
        "title": "Gaussian NMR Boltzmann Averaging App",
        "caption": f"Ver. {APP_VERSION}",
        "description": (
            "Upload Gaussian opt+freq logs and GIAO logs, match conformers by filename, "
            "extract SCF or Gibbs free energies, calculate Boltzmann-averaged isotropic shieldings, "
            "and convert them to chemical shifts using manual references, a TMS log, or linear scaling."
        ),
        "settings": "Settings",
        "developer_info": "Developer information",
        "developer_name": "Name",
        "developer_affiliation": "Affiliation",
        "temperature": "Temperature (K)",
        "energy_mode": "Energy to use for Boltzmann weighting",
        "energy_gibbs": "Gibbs free energy",
        "energy_scf": "SCF energy",
        "shift_mode": "Chemical shift conversion method",
        "shift_manual": "Manual reference shielding",
        "shift_tms": "TMS log file",
        "shift_linear": "Linear scaling",
        "ref_h": "Reference shielding for 1H",
        "ref_c": "Reference shielding for 13C",
        "upload_tms": "Upload TMS GIAO log file",
        "tms_success": "TMS reference extracted successfully.",
        "tms_prev": "Previously loaded TMS reference is being used.",
        "tms_prompt": "Please upload a TMS GIAO log file.",
        "tms_file": "File",
        "tms_h": "TMS 1H reference shielding",
        "tms_c": "TMS 13C reference shielding",
        "slope_h": "Slope for 1H",
        "intercept_h": "Intercept for 1H",
        "slope_c": "Slope for 13C",
        "intercept_c": "Intercept for 13C",
        "element_filter": "Element filter for display",
        "all": "All",
        "h": "H",
        "c": "C",
        "other": "Other",
        "upload_header": "1. Upload files",
        "upload_opt": "Upload opt+freq log files",
        "upload_giao": "Upload GIAO log files",
        "matched_header": "2. Matched conformers",
        "no_valid": "No valid matched conformers were found.",
        "tms_not_ready": "TMS reference has not been loaded yet. Please upload a valid TMS GIAO log file in the sidebar.",
        "weights_header": "3. Energies and Boltzmann weights",
        "shielding_header": "4. Isotropic shielding table for each conformer",
        "avg_header": "5. Per-atom Boltzmann-averaged shielding / shift table",
        "eq_header": "6. Equivalent atom groups",
        "atom_label_header": "Atom label assignment",
        "atom_label_help": "Assign labels to atom indices from the first valid coordinate set. These labels can be used when creating equivalent atom groups.",
        "atom_table_empty": "After valid files are uploaded, the atom list will appear here.",
        "available_atoms": "Available atoms",
        "filter_atoms_group": "Filter atoms for group selection",
        "group_label": "Group label",
        "select_atom_indices": "Select equivalent atom indices",
        "select_atom_labels": "Select equivalent atom labels",
        "group_input_mode": "Group input mode",
        "group_by_structure": "Structure picker",
        "group_by_indices": "Atom indices",
        "group_by_labels": "Atom labels",
        "already_registered": "These atom indices are already included in existing groups",
        "add_group": "Add equivalent atom group",
        "enter_group_label": "Please enter a group label.",
        "select_one_atom": "Please select at least one atom.",
        "registered_groups": "Registered equivalent atom groups",
        "delete": "Delete",
        "eq_avg_header": "7. Equivalent-atom averaged table",
        "download_header": "8. Download outputs",
        "download_per_conf": "Download per-conformer shielding table (CSV)",
        "download_avg": "Download per-atom Boltzmann averaged table (CSV)",
        "download_weights": "Download energy / weight table (CSV)",
        "download_eq": "Download equivalent-atom averaged table (CSV)",
        "settings_io_header": "Settings save / load",
        "download_settings": "Download labels / groups settings (JSON)",
        "upload_settings": "Upload labels / groups settings (JSON)",
        "settings_loaded": "Settings file loaded successfully.",
        "settings_load_error": "Failed to load settings JSON.",
        "clear_groups": "Clear equivalent atom groups",
        "clear_labels": "Clear atom labels",
        "atom_label_column": "atom_label",
        "atom_label_input": "Label",
        "apply_labels": "Apply label edits",
        "labels_updated": "Atom labels were updated.",
        "filename_prefix_info": "Output filenames use the common prefix of uploaded GIAO files.",
        "structure_picker_title": "Structure-based atom picker",
        "structure_picker_help": "Use the 3D viewer to select atoms. The numbering follows the first valid coordinate set and is applied globally.",
        "selected_atoms_preview": "Selected atoms",
        "selected_labels_preview": "Selected labels",
        "expander_show": "Show",
        "summary_columns_note": "The Boltzmann-averaged table includes weight_<conf_id> columns.",
        "coord_not_found": "No coordinate block could be extracted from valid files, so the structure picker is unavailable.",
    },
    "ja": {
        "title": "Gaussian NMR Boltzmann Averaging App",
        "caption": f"Ver. {APP_VERSION}",
        "description": (
            "Gaussian の opt+freq ログと GIAO ログをアップロードし、ファイル名で配座を対応付け、"
            "SCF energy または Gibbs free energy を抽出し、Boltzmann 平均 isotropic shielding を計算し、"
            "手動参照値・TMS ログ・線形補正式を用いて chemical shift に変換します。"
        ),
        "settings": "設定",
        "developer_info": "開発者情報",
        "developer_name": "氏名",
        "developer_affiliation": "所属",
        "temperature": "温度 (K)",
        "energy_mode": "Boltzmann 重み付けに使うエネルギー",
        "energy_gibbs": "Gibbs free energy",
        "energy_scf": "SCF energy",
        "shift_mode": "Chemical shift の変換方法",
        "shift_manual": "手動参照 shielding",
        "shift_tms": "TMS ログファイル",
        "shift_linear": "線形補正式",
        "ref_h": "1H の参照 shielding",
        "ref_c": "13C の参照 shielding",
        "upload_tms": "TMS の GIAO ログをアップロード",
        "tms_success": "TMS 参照値を正常に抽出しました。",
        "tms_prev": "前回読み込んだ TMS 参照値を使用しています。",
        "tms_prompt": "TMS の GIAO ログをアップロードしてください。",
        "tms_file": "ファイル",
        "tms_h": "TMS 1H 参照 shielding",
        "tms_c": "TMS 13C 参照 shielding",
        "slope_h": "1H の slope",
        "intercept_h": "1H の intercept",
        "slope_c": "13C の slope",
        "intercept_c": "13C の intercept",
        "element_filter": "表示元素フィルター",
        "all": "All",
        "h": "H",
        "c": "C",
        "other": "Other",
        "upload_header": "1. ファイルアップロード",
        "upload_opt": "opt+freq ログをアップロード",
        "upload_giao": "GIAO ログをアップロード",
        "matched_header": "2. 対応付けられた配座",
        "no_valid": "有効な対応配座が見つかりませんでした。",
        "tms_not_ready": "TMS 参照値がまだ読み込まれていません。サイドバーから有効な TMS GIAO ログをアップロードしてください。",
        "weights_header": "3. エネルギーと Boltzmann 存在比",
        "shielding_header": "4. 各配座の isotropic shielding テーブル",
        "avg_header": "5. 原子ごとの Boltzmann 平均 shielding / shift テーブル",
        "eq_header": "6. Equivalent atom groups",
        "atom_label_header": "原子ラベル設定",
        "atom_label_help": "最初の有効な座標セットに対してラベルを付与します。付与したラベルは equivalent atom group の作成に利用できます。",
        "atom_table_empty": "有効なファイルがアップロードされると、ここに atom list が表示されます。",
        "available_atoms": "利用可能な原子",
        "filter_atoms_group": "group 選択用の原子フィルター",
        "group_label": "グループラベル",
        "select_atom_indices": "等価原子番号を選択",
        "select_atom_labels": "等価原子ラベルを選択",
        "group_input_mode": "グループ入力モード",
        "group_by_structure": "構造ピッカー",
        "group_by_indices": "原子番号",
        "group_by_labels": "原子ラベル",
        "already_registered": "以下の原子番号は既存グループに含まれています",
        "add_group": "Equivalent atom group を追加",
        "enter_group_label": "グループラベルを入力してください。",
        "select_one_atom": "少なくとも1つ原子を選択してください。",
        "registered_groups": "登録済み equivalent atom groups",
        "delete": "削除",
        "eq_avg_header": "7. Equivalent atom 平均テーブル",
        "download_header": "8. 出力ファイルのダウンロード",
        "download_per_conf": "各配座 shielding テーブル (CSV) をダウンロード",
        "download_avg": "原子ごとの Boltzmann 平均テーブル (CSV) をダウンロード",
        "download_weights": "エネルギー / 存在比テーブル (CSV) をダウンロード",
        "download_eq": "Equivalent atom 平均テーブル (CSV) をダウンロード",
        "settings_io_header": "設定の保存 / 読み込み",
        "download_settings": "ラベル / groups 設定 (JSON) をダウンロード",
        "upload_settings": "ラベル / groups 設定 (JSON) をアップロード",
        "settings_loaded": "設定ファイルを正常に読み込みました。",
        "settings_load_error": "設定 JSON の読み込みに失敗しました。",
        "clear_groups": "Equivalent atom groups をクリア",
        "clear_labels": "原子ラベルをクリア",
        "atom_label_column": "atom_label",
        "atom_label_input": "ラベル",
        "apply_labels": "ラベル編集を反映",
        "labels_updated": "原子ラベルを更新しました。",
        "filename_prefix_info": "出力ファイル名にはアップロードした GIAO ファイルの共通接頭辞を使用します。",
        "structure_picker_title": "構造ベース原子ピッカー",
        "structure_picker_help": "3D ビューアで原子を選択します。番号は最初の有効座標セットに基づき、全体に適用されます。",
        "selected_atoms_preview": "選択原子",
        "selected_labels_preview": "選択ラベル",
        "expander_show": "表示",
        "summary_columns_note": "Boltzmann 平均テーブルには weight_<conf_id> 列を含みます。",
        "coord_not_found": "有効ファイルから座標ブロックを抽出できなかったため、構造ピッカーは使用できません。",
    },
}


if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = "English"
if "equivalent_groups_ui" not in st.session_state:
    st.session_state["equivalent_groups_ui"] = []
if "latest_atom_table" not in st.session_state:
    st.session_state["latest_atom_table"] = pd.DataFrame(columns=["atom_index", "element"])
if "tms_ref_H" not in st.session_state:
    st.session_state["tms_ref_H"] = None
if "tms_ref_C" not in st.session_state:
    st.session_state["tms_ref_C"] = None
if "tms_ref_filename" not in st.session_state:
    st.session_state["tms_ref_filename"] = None
if "atom_label_map" not in st.session_state:
    st.session_state["atom_label_map"] = {}
if "settings_loaded_once" not in st.session_state:
    st.session_state["settings_loaded_once"] = False
if "picker_selected_atoms" not in st.session_state:
    st.session_state["picker_selected_atoms"] = []
if "latest_xyz" not in st.session_state:
    st.session_state["latest_xyz"] = ""


def get_lang():
    return "ja" if st.session_state["ui_language"] == "日本語" else "en"


def T(key):
    return UI_TEXT[get_lang()][key]


def extract_conf_id(filename: str):
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


def extract_last_xyz_from_gaussian(text: str):
    lines = text.splitlines()
    blocks = []

    for i, line in enumerate(lines):
        if "Standard orientation:" in line or "Input orientation:" in line:
            start = i + 5
            rows = []
            j = start
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    break
                if s.startswith("-----"):
                    break

                parts = lines[j].split()
                if len(parts) >= 6:
                    try:
                        atomic_num = int(parts[1])
                        x = float(parts[3])
                        y = float(parts[4])
                        z = float(parts[5])
                        rows.append((atomic_num, x, y, z))
                    except Exception:
                        pass
                j += 1

            if rows:
                blocks.append(rows)

    if not blocks:
        return ""

    last = blocks[-1]
    xyz_lines = [str(len(last)), "Gaussian coordinates"]
    for atomic_num, x, y, z in last:
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_num, "X")
        xyz_lines.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(xyz_lines)


def get_tms_reference_from_log(text):
    df = extract_isotropic_shieldings(text)

    if df.empty:
        return None, None, "No isotropic shielding entries were found in the TMS log."
    if not check_normal_termination(text):
        return None, None, "The TMS log did not terminate normally."

    h_df = df[df["element"] == "H"].copy()
    c_df = df[df["element"] == "C"].copy()

    if h_df.empty:
        return None, None, "No hydrogen shielding values were found in the TMS log."
    if c_df.empty:
        return None, None, "No carbon shielding values were found in the TMS log."

    return h_df["shielding"].mean(), c_df["shielding"].mean(), None


def boltzmann_weights(energies_hartree, temperature=298.15):
    energies_hartree = np.array(energies_hartree, dtype=float)
    rel_kcal = (energies_hartree - energies_hartree.min()) * HARTREE_TO_KCAL
    weights = np.exp(-rel_kcal / (R_KCAL * temperature))
    weights /= weights.sum()
    return rel_kcal, weights


def build_per_conformer_shielding_table(shielding_map, conf_ids):
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
        weight_col = f"weight_{cid}"
        out[weighted_col] = out[col] * w
        out[weight_col] = w

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
    results = []

    value_cols = [
        c for c in df.columns
        if c.startswith("shielding_") or c.startswith("weighted_") or c.startswith("weight_")
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
            "atom_labels": ",".join(group.get("atom_labels", [])),
            "n_atoms": len(atoms),
            "element": element_label,
            "input_mode": group.get("input_mode", "atom_indices"),
        }

        for col in value_cols:
            row[col] = sub[col].mean()

        results.append(row)

    if results:
        return pd.DataFrame(results)

    return pd.DataFrame(columns=["group_label", "atom_indices", "atom_labels", "n_atoms", "element", "input_mode"])


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
        atom_label = row.get("atom_label", "")
        if pd.notna(atom_label) and str(atom_label).strip():
            labels.append(f'{row["atom_index"]} ({row["element"]}) [{atom_label}]')
        else:
            labels.append(f'{row["atom_index"]} ({row["element"]})')
    return labels


def parse_atom_indices_from_labels(labels):
    atoms = []
    for label in labels:
        m = re.match(r"^\s*(\d+)", str(label))
        if m:
            atoms.append(int(m.group(1)))
    return sorted(set(atoms))


def update_atom_table_with_labels(atom_df, atom_label_map):
    out = atom_df.copy()
    if out.empty:
        out["atom_label"] = []
        return out
    out["atom_label"] = out["atom_index"].map(lambda x: atom_label_map.get(str(int(x)), ""))
    return out


def sanitize_filename_part(text):
    text = Path(text).stem
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "output"


def longest_common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        max_len = min(len(prefix), len(s))
        while i < max_len and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def clean_common_prefix(prefix):
    prefix = prefix.strip("_.- ")
    prefix = re.sub(r"[_\-.]+$", "", prefix)
    return prefix


def build_output_prefix_from_giao(giao_files, min_prefix_len=3, fallback="output"):
    stems = [sanitize_filename_part(f.name) for f in giao_files] if giao_files else []
    if not stems:
        return fallback
    if len(stems) == 1:
        return stems[0]

    prefix = clean_common_prefix(longest_common_prefix(stems))
    if len(prefix) >= min_prefix_len:
        return prefix
    return fallback


def get_label_to_atom_map(atom_label_map):
    label_to_atom = {}
    for k, v in atom_label_map.items():
        label = str(v).strip()
        if label:
            label_to_atom[label] = int(k)
    return label_to_atom


def make_settings_json_bytes():
    payload = {
        "app": "Gaussian NMR Boltzmann Averaging App",
        "version": APP_VERSION,
        "atom_label_map": st.session_state.get("atom_label_map", {}),
        "equivalent_groups_ui": st.session_state.get("equivalent_groups_ui", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_settings_json(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    data = json.loads(text)

    atom_label_map = data.get("atom_label_map", {})
    equivalent_groups_ui = data.get("equivalent_groups_ui", [])

    if not isinstance(atom_label_map, dict):
        raise ValueError("atom_label_map is not a dictionary.")
    if not isinstance(equivalent_groups_ui, list):
        raise ValueError("equivalent_groups_ui is not a list.")

    normalized_map = {}
    for k, v in atom_label_map.items():
        normalized_map[str(int(k))] = str(v)

    normalized_groups = []
    for g in equivalent_groups_ui:
        if not isinstance(g, dict):
            continue
        atoms = sorted(set(int(x) for x in g.get("atoms", [])))
        label = str(g.get("label", "")).strip()
        atom_labels = [str(x) for x in g.get("atom_labels", []) if str(x).strip()]
        input_mode = str(g.get("input_mode", "atom_indices"))
        if label and atoms:
            normalized_groups.append(
                {
                    "label": label,
                    "atoms": atoms,
                    "atom_labels": atom_labels,
                    "input_mode": input_mode,
                }
            )
    return normalized_map, normalized_groups


with st.sidebar:
    selected_language = st.selectbox(
        "Language / 言語",
        options=["English", "日本語"],
        index=0 if st.session_state["ui_language"] == "English" else 1,
    )
    st.session_state["ui_language"] = selected_language

st.title(T("title"))
st.caption(T("caption"))
st.write(T("description"))

st.sidebar.header(T("settings"))
with st.sidebar.expander(T("developer_info"), expanded=False):
    st.sidebar.write(f'**{T("developer_name")}**: {DEVELOPER_INFO["name"]}')
    if get_lang() == "ja":
        st.sidebar.write(f'**{T("developer_affiliation")}**: {DEVELOPER_INFO["affiliation_ja"]}')
    else:
        st.sidebar.write(f'**{T("developer_affiliation")}**: {DEVELOPER_INFO["affiliation_en"]}')

temperature = st.sidebar.number_input(T("temperature"), value=298.15, step=1.0)

energy_mode = st.sidebar.radio(
    T("energy_mode"),
    [T("energy_gibbs"), T("energy_scf")],
    index=0,
)

shift_mode = st.sidebar.radio(
    T("shift_mode"),
    [T("shift_manual"), T("shift_tms"), T("shift_linear")],
    index=0,
)

ref_H = None
ref_C = None
slope_H = None
intercept_H = None
slope_C = None
intercept_C = None

if shift_mode == T("shift_manual"):
    ref_H = st.sidebar.number_input(T("ref_h"), value=31.5)
    ref_C = st.sidebar.number_input(T("ref_c"), value=185.0)

elif shift_mode == T("shift_tms"):
    tms_file = st.sidebar.file_uploader(
        T("upload_tms"),
        type=["log", "out"],
        accept_multiple_files=False,
        key="tms_log",
    )

    if tms_file is not None:
        tms_text = read_text(tms_file)
        parsed_ref_H, parsed_ref_C, tms_error = get_tms_reference_from_log(tms_text)

        if tms_error:
            st.session_state["tms_ref_H"] = None
            st.session_state["tms_ref_C"] = None
            st.session_state["tms_ref_filename"] = None
            st.sidebar.error(tms_error)
        else:
            st.session_state["tms_ref_H"] = parsed_ref_H
            st.session_state["tms_ref_C"] = parsed_ref_C
            st.session_state["tms_ref_filename"] = tms_file.name
            st.sidebar.success(T("tms_success"))
            st.sidebar.write(f'{T("tms_file")}: {tms_file.name}')
            st.sidebar.write(f'{T("tms_h")}: {parsed_ref_H:.4f}')
            st.sidebar.write(f'{T("tms_c")}: {parsed_ref_C:.4f}')

    elif st.session_state["tms_ref_H"] is not None and st.session_state["tms_ref_C"] is not None:
        st.sidebar.success(T("tms_prev"))
        if st.session_state["tms_ref_filename"]:
            st.sidebar.write(f'{T("tms_file")}: {st.session_state["tms_ref_filename"]}')
        st.sidebar.write(f'{T("tms_h")}: {st.session_state["tms_ref_H"]:.4f}')
        st.sidebar.write(f'{T("tms_c")}: {st.session_state["tms_ref_C"]:.4f}')
    else:
        st.sidebar.info(T("tms_prompt"))

    ref_H = st.session_state["tms_ref_H"]
    ref_C = st.session_state["tms_ref_C"]

elif shift_mode == T("shift_linear"):
    slope_H = st.sidebar.number_input(T("slope_h"), value=1.0)
    intercept_H = st.sidebar.number_input(T("intercept_h"), value=31.5)
    slope_C = st.sidebar.number_input(T("slope_c"), value=1.0)
    intercept_C = st.sidebar.number_input(T("intercept_c"), value=185.0)

element_filter = st.sidebar.selectbox(
    T("element_filter"),
    [T("all"), T("h"), T("c"), T("other")],
    index=0,
)

st.subheader(T("upload_header"))

opt_files = st.file_uploader(
    T("upload_opt"),
    type=["log", "out"],
    accept_multiple_files=True,
    key="opt_files",
)

giao_files = st.file_uploader(
    T("upload_giao"),
    type=["log", "out"],
    accept_multiple_files=True,
    key="giao_files",
)

result_df = None
per_conf_df = None
valid_df = None
eq_df = None
output_prefix = build_output_prefix_from_giao(giao_files)

if giao_files:
    st.caption(T("filename_prefix_info"))

st.subheader(T("settings_io_header"))
col_set1, col_set2, col_set3 = st.columns([2, 2, 1])

with col_set1:
    st.download_button(
        label=T("download_settings"),
        data=make_settings_json_bytes(),
        file_name=f"{output_prefix}_nmr_labels_groups_settings.json",
        mime="application/json",
    )

with col_set2:
    settings_file = st.file_uploader(
        T("upload_settings"),
        type=["json"],
        accept_multiple_files=False,
        key="settings_json",
    )
    if settings_file is not None and not st.session_state["settings_loaded_once"]:
        try:
            atom_label_map_loaded, groups_loaded = load_settings_json(settings_file)
            st.session_state["atom_label_map"] = atom_label_map_loaded
            st.session_state["equivalent_groups_ui"] = groups_loaded
            st.session_state["settings_loaded_once"] = True
            st.success(T("settings_loaded"))
            st.rerun()
        except Exception:
            st.error(T("settings_load_error"))
    if settings_file is None:
        st.session_state["settings_loaded_once"] = False

with col_set3:
    if st.button(T("clear_groups")):
        st.session_state["equivalent_groups_ui"] = []
        st.rerun()
    if st.button(T("clear_labels")):
        st.session_state["atom_label_map"] = {}
        st.rerun()

if opt_files and giao_files:
    opt_records = []
    xyz_candidates = []

    for f in opt_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        gibbs = extract_gibbs_free_energy(text)
        scf = extract_last_scf_energy(text)
        normal = check_normal_termination(text)
        xyz_text = extract_last_xyz_from_gaussian(text)
        if xyz_text:
            xyz_candidates.append(xyz_text)

        opt_records.append(
            {
                "conf_id": cid,
                "opt_filename": f.name,
                "gibbs_hartree": gibbs,
                "scf_hartree": scf,
                "opt_normal_termination": normal,
                "xyz_text": xyz_text,
            }
        )

    opt_df = pd.DataFrame(opt_records)

    giao_records = []
    shielding_map = {}

    for f in giao_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        normal = check_normal_termination(text)
        shielding_df = extract_isotropic_shieldings(text)
        xyz_text = extract_last_xyz_from_gaussian(text)
        if xyz_text:
            xyz_candidates.append(xyz_text)

        giao_records.append(
            {
                "conf_id": cid,
                "giao_filename": f.name,
                "n_atoms_found": len(shielding_df),
                "giao_normal_termination": normal,
                "giao_xyz_text": xyz_text,
            }
        )
        shielding_map[cid] = shielding_df

    giao_df = pd.DataFrame(giao_records)
    pair_df = pd.merge(opt_df, giao_df, on="conf_id", how="inner")

    if energy_mode == T("energy_gibbs"):
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
        st.error(T("no_valid"))
        st.stop()

    tms_ready = True
    if shift_mode == T("shift_tms") and (ref_H is None or ref_C is None):
        tms_ready = False
        st.warning(T("tms_not_ready"))

    rel_kcal, weights = boltzmann_weights(valid_df[energy_col].values, temperature=temperature)
    valid_df["energy_used_hartree"] = valid_df[energy_col]
    valid_df["relative_energy_kcal"] = rel_kcal
    valid_df["boltzmann_weight"] = weights

    conf_ids = valid_df["conf_id"].tolist()
    per_conf_df_full = build_per_conformer_shielding_table(shielding_map, conf_ids)

    atom_table_full = (
        per_conf_df_full[["atom_index", "element"]]
        .drop_duplicates()
        .sort_values("atom_index")
        .reset_index(drop=True)
    )
    atom_table_full = update_atom_table_with_labels(atom_table_full, st.session_state["atom_label_map"])
    st.session_state["latest_atom_table"] = atom_table_full.copy()

    valid_xyz = ""
    for _, row in valid_df.iterrows():
        if isinstance(row.get("xyz_text"), str) and row.get("xyz_text").strip():
            valid_xyz = row["xyz_text"]
            break
        if isinstance(row.get("giao_xyz_text"), str) and row.get("giao_xyz_text").strip():
            valid_xyz = row["giao_xyz_text"]
            break
    if not valid_xyz and xyz_candidates:
        valid_xyz = xyz_candidates[0]
    st.session_state["latest_xyz"] = valid_xyz

    per_conf_df = per_conf_df_full.copy()
    if element_filter == T("h"):
        per_conf_df = per_conf_df[per_conf_df["element"] == "H"].copy()
    elif element_filter == T("c"):
        per_conf_df = per_conf_df[per_conf_df["element"] == "C"].copy()
    elif element_filter == T("other"):
        per_conf_df = per_conf_df[~per_conf_df["element"].isin(["H", "C"])].copy()

    per_conf_df = pd.merge(
        per_conf_df,
        atom_table_full[["atom_index", "element", "atom_label"]],
        on=["atom_index", "element"],
        how="left",
    )
    cols_front = ["atom_index", "element", "atom_label"]
    other_cols = [c for c in per_conf_df.columns if c not in cols_front]
    per_conf_df = per_conf_df[cols_front + other_cols]

    avg_df = add_boltzmann_average(per_conf_df, conf_ids, weights)

    if shift_mode == T("shift_manual"):
        result_df = shielding_to_shift(
            avg_df,
            mode="manual_reference",
            ref_H=ref_H,
            ref_C=ref_C,
        )
    elif shift_mode == T("shift_tms"):
        if tms_ready:
            result_df = shielding_to_shift(
                avg_df,
                mode="tms_log",
                ref_H=ref_H,
                ref_C=ref_C,
            )
        else:
            result_df = avg_df.copy()
            result_df["chemical_shift"] = np.nan
    else:
        result_df = shielding_to_shift(
            avg_df,
            mode="linear",
            slope_H=slope_H,
            intercept_H=intercept_H,
            slope_C=slope_C,
            intercept_C=intercept_C,
        )

    with st.expander(T("matched_header"), expanded=False):
        st.dataframe(pair_df, use_container_width=True)

    with st.expander(T("weights_header"), expanded=False):
        st.dataframe(valid_df, use_container_width=True)

    with st.expander(T("shielding_header"), expanded=False):
        st.dataframe(per_conf_df, use_container_width=True)

    with st.expander(T("avg_header"), expanded=False):
        st.caption(T("summary_columns_note"))
        st.dataframe(result_df, use_container_width=True)

st.subheader(T("eq_header"))

atom_df_ui = st.session_state["latest_atom_table"].copy()

if atom_df_ui.empty:
    st.info(T("atom_table_empty"))
else:
    st.markdown(f"**{T('atom_label_header')}**")
    st.caption(T("atom_label_help"))

    label_editor_df = atom_df_ui[["atom_index", "element", "atom_label"]].copy()
    label_editor_df = label_editor_df.rename(columns={"atom_label": T("atom_label_column")})

    edited_labels_df = st.data_editor(
        label_editor_df,
        use_container_width=True,
        num_rows="fixed",
        disabled=["atom_index", "element"],
        key="atom_label_editor_widget",
    )

    if st.button(T("apply_labels"), key="apply_labels_button"):
        new_map = {}
        label_col = T("atom_label_column")
        for _, row in edited_labels_df.iterrows():
            label = str(row[label_col]).strip() if pd.notna(row[label_col]) else ""
            if label:
                new_map[str(int(row["atom_index"]))] = label
        st.session_state["atom_label_map"] = new_map
        st.success(T("labels_updated"))
        st.rerun()

    ui_filter = st.selectbox(
        T("filter_atoms_group"),
        [T("all"), T("h"), T("c"), T("other")],
        index=0,
        key="eq_ui_filter",
    )

    atom_df_filtered = atom_df_ui.copy()
    if ui_filter == T("h"):
        atom_df_filtered = atom_df_filtered[atom_df_filtered["element"] == "H"].copy()
    elif ui_filter == T("c"):
        atom_df_filtered = atom_df_filtered[atom_df_filtered["element"] == "C"].copy()
    elif ui_filter == T("other"):
        atom_df_filtered = atom_df_filtered[~atom_df_filtered["element"].isin(["H", "C"])].copy()

    registered_atoms = get_registered_atom_set(st.session_state["equivalent_groups_ui"])
    atom_df_filtered["already_registered"] = atom_df_filtered["atom_index"].isin(registered_atoms)

    st.write(T("available_atoms"))
    st.dataframe(atom_df_filtered, use_container_width=True)

    group_label = st.text_input(T("group_label"), value="", key="eq_group_label")

    group_input_mode = st.radio(
        T("group_input_mode"),
        [T("group_by_structure"), T("group_by_indices"), T("group_by_labels")],
        index=0,
        key="eq_group_input_mode",
    )

    selected_atoms = []
    selected_atom_labels = []

    if group_input_mode == T("group_by_structure"):
        st.markdown(f"**{T('structure_picker_title')}**")
        if st.session_state["latest_xyz"]:
            st.caption(T("structure_picker_help"))
            picker_default = st.session_state.get("picker_selected_atoms", [])
            picker_value = ATOM_PICKER(
                xyz=st.session_state["latest_xyz"],
                selected_atoms=picker_default,
                language=get_lang(),
                height=520,
                key="atom_picker_component_instance",
                default=picker_default,
            )
            if picker_value is None:
                picker_value = picker_default
            st.session_state["picker_selected_atoms"] = sorted(set(int(x) for x in picker_value))
            selected_atoms = st.session_state["picker_selected_atoms"]

            reverse_map = {
                int(k): v.strip()
                for k, v in st.session_state["atom_label_map"].items()
                if str(v).strip()
            }
            selected_atom_labels = [reverse_map[a] for a in selected_atoms if a in reverse_map]

            st.write(f"**{T('selected_atoms_preview')}**: " + (", ".join(map(str, selected_atoms)) if selected_atoms else "-"))
            st.write(f"**{T('selected_labels_preview')}**: " + (", ".join(selected_atom_labels) if selected_atom_labels else "-"))
        else:
            st.info(T("coord_not_found"))

    elif group_input_mode == T("group_by_indices"):
        atom_option_labels = make_atom_option_labels(atom_df_filtered[["atom_index", "element", "atom_label"]])
        selected_atom_labels_raw = st.multiselect(
            T("select_atom_indices"),
            options=atom_option_labels,
            default=[],
            key="eq_selected_atoms_indices",
        )
        selected_atoms = parse_atom_indices_from_labels(selected_atom_labels_raw)

        reverse_map = {
            int(k): v.strip()
            for k, v in st.session_state["atom_label_map"].items()
            if str(v).strip()
        }
        selected_atom_labels = [reverse_map[a] for a in selected_atoms if a in reverse_map]

    else:
        label_to_atom = get_label_to_atom_map(st.session_state["atom_label_map"])
        available_label_to_atom = {
            label: atom
            for label, atom in label_to_atom.items()
            if atom in atom_df_filtered["atom_index"].tolist()
        }
        selected_atom_labels = st.multiselect(
            T("select_atom_labels"),
            options=sorted(available_label_to_atom.keys()),
            default=[],
            key="eq_selected_atoms_labels",
        )
        selected_atoms = sorted(set(available_label_to_atom[label] for label in selected_atom_labels))

    if selected_atoms:
        overlapping_atoms = sorted(set(selected_atoms) & registered_atoms)
        if overlapping_atoms:
            st.warning(T("already_registered") + ": " + ", ".join(map(str, overlapping_atoms)))

    if st.button(T("add_group"), key="add_eq_group"):
        clean_label = group_label.strip()

        if not clean_label:
            st.warning(T("enter_group_label"))
        elif not selected_atoms:
            st.warning(T("select_one_atom"))
        else:
            st.session_state["equivalent_groups_ui"].append(
                {
                    "label": clean_label,
                    "atoms": selected_atoms,
                    "atom_labels": selected_atom_labels,
                    "input_mode": (
                        "structure_picker" if group_input_mode == T("group_by_structure")
                        else "atom_labels" if group_input_mode == T("group_by_labels")
                        else "atom_indices"
                    ),
                }
            )
            st.rerun()

if st.session_state["equivalent_groups_ui"]:
    st.write(T("registered_groups"))

    registered_rows = []
    for g in st.session_state["equivalent_groups_ui"]:
        registered_rows.append(
            {
                "label": g["label"],
                "atom_indices": ", ".join(map(str, g["atoms"])),
                "atom_labels": ", ".join(g.get("atom_labels", [])),
                "n_atoms": len(g["atoms"]),
                "input_mode": g.get("input_mode", ""),
            }
        )
    st.dataframe(pd.DataFrame(registered_rows), use_container_width=True)

    for i, g in enumerate(st.session_state["equivalent_groups_ui"]):
        col1, col2, col3, col4 = st.columns([2, 4, 3, 1])
        col1.write(f'**{g["label"]}**')
        col2.write(", ".join(map(str, g["atoms"])))
        col3.write(", ".join(g.get("atom_labels", [])))
        if col4.button(T("delete"), key=f"delete_group_{i}"):
            st.session_state["equivalent_groups_ui"].pop(i)
            st.rerun()

if result_df is not None:
    if st.session_state["equivalent_groups_ui"]:
        eq_df = average_equivalent_atoms(result_df, st.session_state["equivalent_groups_ui"])
        st.subheader(T("eq_avg_header"))
        st.dataframe(eq_df, use_container_width=True)

    st.subheader(T("download_header"))

    if per_conf_df is not None:
        st.download_button(
            label=T("download_per_conf"),
            data=dataframe_to_csv_bytes(per_conf_df),
            file_name=f"{output_prefix}_per_conformer_isotropic_shieldings.csv",
            mime="text/csv",
        )

    st.download_button(
        label=T("download_avg"),
        data=dataframe_to_csv_bytes(result_df),
        file_name=f"{output_prefix}_boltzmann_averaged_nmr.csv",
        mime="text/csv",
    )

    if valid_df is not None:
        st.download_button(
            label=T("download_weights"),
            data=dataframe_to_csv_bytes(valid_df),
            file_name=f"{output_prefix}_boltzmann_weights.csv",
            mime="text/csv",
        )

    if eq_df is not None:
        st.download_button(
            label=T("download_eq"),
            data=dataframe_to_csv_bytes(eq_df),
            file_name=f"{output_prefix}_equivalent_atom_averaged_nmr.csv",
            mime="text/csv",
        )
