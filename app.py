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

APP_VERSION = "2.2"

DEVELOPER_INFO = {
    "name": "Ken-ichi Nakashima",
    "affiliation_ja": "愛知学院大学 薬学部 薬用資源学講座",
    "affiliation_en": "Aichi-Gakuin University, School of Pharmacy, Laboratory of Natural Resources",
}

_COMPONENT_DIR = Path(__file__).parent / "atom_picker_component"
atom_picker_component = components.declare_component(
    "atom_picker_component",
    path=str(_COMPONENT_DIR),
)

ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 2: "He",
    3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 53: "I",
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
        "mapping_header": "6. Atom mapping",
        "mapping_desc": "Click one or more atoms in the 3D structure, assign a proton label, and save the mapping. Multiple selected atoms are treated as an interchangeable group.",
        "click_atoms": "#### Click atoms in the 3D structure",
        "viewer_caption": "Hydrogen atom numbers are shown by default. Use the checkboxes above the viewer to display all atom numbers or to show hydrogens only.",
        "new_mapping": "#### New proton mapping",
        "selected": "Selected: {items}",
        "none_selected": "No atom is selected yet.",
        "label_entry": "Label entry",
        "build_label": "Build H label",
        "free_text": "Free text",
        "position_number": "Position number",
        "position_placeholder": "e.g. 3",
        "prime": "Prime",
        "display_label": "Display label: **{label}**",
        "label": "Label",
        "label_placeholder": "e.g. H-2′ / H-6′",
        "save_mapping": "Save mapping",
        "clear_selection": "Clear selection",
        "enter_label": "Enter a proton label.",
        "select_atom": "Select at least one atom in the 3D structure.",
        "non_hydrogen": "The following selected atoms are not hydrogen atoms: {atoms}",
        "added": "Added {label}.",
        "updated": "Updated {label}.",
        "manual_fallback": "Manual selection fallback",
        "manual_desc": "Use this only when clicking the 3D structure is difficult.",
        "manual_numbers": "1-based atom numbers, separated by commas",
        "apply_manual": "Apply manual selection",
        "out_of_range": "Out-of-range atom numbers: {atoms}",
        "integer_error": "Enter integers separated by commas.",
        "registered_mappings": "#### Registered mappings",
        "no_mappings": "No mappings have been registered.",
        "not_assigned": "Not assigned",
        "atoms_label": "Atom(s)",
        "show_edit": "Show/edit atoms",
        "delete": "Delete",
        "mapping_status": "Mapping status",
        "mapping_labels": "Labels",
        "mapping_registered": "Registered labels",
        "eq_avg_header": "7. Equivalent-atom averaged table",
        "download_header": "8. Download outputs",
        "download_per_conf": "Download per-conformer shielding table (CSV)",
        "download_avg": "Download per-atom Boltzmann averaged table (CSV)",
        "download_weights": "Download energy / weight table (CSV)",
        "download_eq": "Download equivalent-atom averaged table (CSV)",
        "settings_io_header": "Settings save / load",
        "download_settings": "Download mapping settings (JSON)",
        "upload_settings": "Upload mapping settings (JSON)",
        "settings_loaded": "Settings file loaded successfully.",
        "settings_load_error": "Failed to load settings JSON.",
        "clear_mappings": "Clear mappings",
        "filename_prefix_info": "Output filenames use the common prefix of uploaded GIAO files.",
        "summary_columns_note": "The Boltzmann-averaged table includes weight_<conf_id> columns.",
        "coord_not_found": "No coordinate block could be extracted from valid files, so the structure picker is unavailable.",
        "save_project_desc": "The JSON file stores only proton labels / equivalent groups, not the Gaussian logs or calculation results.",
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
        "mapping_header": "6. Atom mapping",
        "mapping_desc": "3D 構造上で1つ以上の原子を選択し、プロトンラベルを付けて保存します。複数選択した原子は交換可能なグループとして扱います。",
        "click_atoms": "#### 3D 構造上で原子をクリック",
        "viewer_caption": "初期状態では水素原子番号が表示されます。上部のチェックボックスで全原子番号表示や水素のみ表示に切り替えられます。",
        "new_mapping": "#### 新しいプロトンマッピング",
        "selected": "選択中: {items}",
        "none_selected": "まだ原子が選択されていません。",
        "label_entry": "ラベル入力",
        "build_label": "Hラベルを組み立て",
        "free_text": "自由入力",
        "position_number": "位置番号",
        "position_placeholder": "例: 3",
        "prime": "Prime",
        "display_label": "表示ラベル: **{label}**",
        "label": "ラベル",
        "label_placeholder": "例: H-2′ / H-6′",
        "save_mapping": "マッピングを保存",
        "clear_selection": "選択をクリア",
        "enter_label": "プロトンラベルを入力してください。",
        "select_atom": "少なくとも1つ原子を選択してください。",
        "non_hydrogen": "以下の選択原子は水素ではありません: {atoms}",
        "added": "{label} を追加しました。",
        "updated": "{label} を更新しました。",
        "manual_fallback": "手動選択フォールバック",
        "manual_desc": "3D 構造上でのクリックが難しい場合にのみ使ってください。",
        "manual_numbers": "1始まりの原子番号をカンマ区切りで入力",
        "apply_manual": "手動選択を適用",
        "out_of_range": "範囲外の原子番号: {atoms}",
        "integer_error": "整数をカンマ区切りで入力してください。",
        "registered_mappings": "#### 登録済みマッピング",
        "no_mappings": "まだマッピングは登録されていません。",
        "not_assigned": "未割り当て",
        "atoms_label": "原子",
        "show_edit": "原子を表示/編集",
        "delete": "削除",
        "mapping_status": "マッピング状況",
        "mapping_labels": "ラベル",
        "mapping_registered": "登録数",
        "eq_avg_header": "7. Equivalent atom 平均テーブル",
        "download_header": "8. 出力ファイルのダウンロード",
        "download_per_conf": "各配座 shielding テーブル (CSV) をダウンロード",
        "download_avg": "原子ごとの Boltzmann 平均テーブル (CSV) をダウンロード",
        "download_weights": "エネルギー / 存在比テーブル (CSV) をダウンロード",
        "download_eq": "Equivalent atom 平均テーブル (CSV) をダウンロード",
        "settings_io_header": "設定の保存 / 読み込み",
        "download_settings": "マッピング設定 (JSON) をダウンロード",
        "upload_settings": "マッピング設定 (JSON) をアップロード",
        "settings_loaded": "設定ファイルを正常に読み込みました。",
        "settings_load_error": "設定 JSON の読み込みに失敗しました。",
        "clear_mappings": "マッピングをクリア",
        "filename_prefix_info": "出力ファイル名にはアップロードした GIAO ファイルの共通接頭辞を使用します。",
        "summary_columns_note": "Boltzmann 平均テーブルには weight_<conf_id> 列を含みます。",
        "coord_not_found": "有効ファイルから座標ブロックを抽出できなかったため、構造ピッカーは使用できません。",
        "save_project_desc": "JSON ファイルにはプロトンラベル / equivalent group の情報のみを保存し、Gaussian ログや計算結果は保存しません。",
    },
}


def current_language():
    return "ja" if st.session_state.get("ui_language", "English") == "日本語" else "en"


def t(key: str, **kwargs):
    text = UI_TEXT[current_language()].get(key, key)
    return text.format(**kwargs) if kwargs else text


if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = "English"
if "tms_ref_H" not in st.session_state:
    st.session_state["tms_ref_H"] = None
if "tms_ref_C" not in st.session_state:
    st.session_state["tms_ref_C"] = None
if "tms_ref_filename" not in st.session_state:
    st.session_state["tms_ref_filename"] = None
if "latest_atom_table" not in st.session_state:
    st.session_state["latest_atom_table"] = pd.DataFrame(columns=["atom_index", "element"])
if "latest_xyz" not in st.session_state:
    st.session_state["latest_xyz"] = ""
if "atom_mappings" not in st.session_state:
    st.session_state["atom_mappings"] = []
if "mapping_selection" not in st.session_state:
    st.session_state["mapping_selection"] = []
if "settings_loaded_once" not in st.session_state:
    st.session_state["settings_loaded_once"] = False


def atom_picker(atoms, xyz_text, selected_atoms=None, height=520, language="ja", key=None):
    default_selection = sorted(set(int(x) for x in (selected_atoms or [])))
    value = atom_picker_component(
        xyz=xyz_text,
        selected_atoms=default_selection,
        height=int(height),
        language=str(language),
        key=key,
        default=default_selection,
    )
    if not isinstance(value, list):
        return default_selection

    cleaned = []
    for item in value:
        try:
            number = int(item)
            if 1 <= number <= len(atoms):
                cleaned.append(number)
        except Exception:
            pass
    return sorted(set(cleaned))


def read_text(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def extract_conf_id(filename: str):
    m = re.search(r"(\d+)\.(log|out)$", filename, re.IGNORECASE)
    if m:
        return m.group(1)

    m2 = re.search(r"conf[_\- ]*(\d+)", filename, re.IGNORECASE)
    if m2:
        return m2.group(1)

    stem = re.sub(r"\.(log|out)$", "", filename, flags=re.IGNORECASE)
    return stem


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
                if not s or s.startswith("-----"):
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
        shielding_col = f"shielding_{cid}"
        weighted_col = f"weighted_{cid}"
        weight_col = f"weight_{cid}"

        out[weighted_col] = out[shielding_col] * w
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


def average_equivalent_atoms(df, mappings):
    results = []

    value_cols = [c for c in df.columns if c.startswith("shielding_") or c.startswith("weighted_") or c.startswith("weight_")]
    if "shielding_boltzmann" in df.columns:
        value_cols.append("shielding_boltzmann")
    if "chemical_shift" in df.columns:
        value_cols.append("chemical_shift")
    value_cols = list(dict.fromkeys(value_cols))

    for item in mappings:
        atom_numbers = [int(n) for n in item.get("atom_numbers", [])]
        sub = df[df["atom_index"].isin(atom_numbers)].copy()
        if sub.empty:
            continue

        elements = sorted(sub["element"].dropna().unique().tolist())
        element_label = "/".join(elements) if elements else ""

        row = {
            "group_label": item["label"],
            "atom_indices": ",".join(map(str, atom_numbers)),
            "n_atoms": len(atom_numbers),
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


def atom_index_from_user_number(user_number: int):
    return user_number - 1


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


def make_settings_json_bytes():
    payload = {
        "app": "Gaussian NMR Boltzmann Averaging App",
        "version": APP_VERSION,
        "atom_mappings": st.session_state.get("atom_mappings", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_settings_json(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    data = json.loads(text)

    mappings = data.get("atom_mappings", [])
    if not isinstance(mappings, list):
        raise ValueError("atom_mappings is not a list.")

    normalized = []
    for item in mappings:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        atom_numbers = sorted(set(int(x) for x in item.get("atom_numbers", [])))
        atom_indices = [atom_index_from_user_number(n) for n in atom_numbers]
        if label and atom_numbers:
            normalized.append(
                {
                    "label": label,
                    "atom_numbers": atom_numbers,
                    "atom_indices": atom_indices,
                }
            )
    return normalized


with st.sidebar:
    selected_language = st.selectbox(
        "Language / 言語",
        options=["English", "日本語"],
        index=0 if st.session_state["ui_language"] == "English" else 1,
    )
    st.session_state["ui_language"] = selected_language

st.title(t("title"))
st.caption(t("caption"))
st.write(t("description"))

st.sidebar.header(t("settings"))
with st.sidebar.expander(t("developer_info"), expanded=False):
    st.sidebar.write(f'**{t("developer_name")}**: {DEVELOPER_INFO["name"]}')
    if current_language() == "ja":
        st.sidebar.write(f'**{t("developer_affiliation")}**: {DEVELOPER_INFO["affiliation_ja"]}')
    else:
        st.sidebar.write(f'**{t("developer_affiliation")}**: {DEVELOPER_INFO["affiliation_en"]}')

temperature = st.sidebar.number_input(t("temperature"), value=298.15, step=1.0)

energy_mode = st.sidebar.radio(
    t("energy_mode"),
    [t("energy_gibbs"), t("energy_scf")],
    index=0,
)

shift_mode = st.sidebar.radio(
    t("shift_mode"),
    [t("shift_manual"), t("shift_tms"), t("shift_linear")],
    index=0,
)

ref_H = None
ref_C = None
slope_H = None
intercept_H = None
slope_C = None
intercept_C = None

if shift_mode == t("shift_manual"):
    ref_H = st.sidebar.number_input(t("ref_h"), value=31.5)
    ref_C = st.sidebar.number_input(t("ref_c"), value=185.0)

elif shift_mode == t("shift_tms"):
    tms_file = st.sidebar.file_uploader(
        t("upload_tms"),
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
            st.sidebar.success(t("tms_success"))
            st.sidebar.write(f'{t("tms_file")}: {tms_file.name}')
            st.sidebar.write(f'{t("tms_h")}: {parsed_ref_H:.4f}')
            st.sidebar.write(f'{t("tms_c")}: {parsed_ref_C:.4f}')

    elif st.session_state["tms_ref_H"] is not None and st.session_state["tms_ref_C"] is not None:
        st.sidebar.success(t("tms_prev"))
        if st.session_state["tms_ref_filename"]:
            st.sidebar.write(f'{t("tms_file")}: {st.session_state["tms_ref_filename"]}')
        st.sidebar.write(f'{t("tms_h")}: {st.session_state["tms_ref_H"]:.4f}')
        st.sidebar.write(f'{t("tms_c")}: {st.session_state["tms_ref_C"]:.4f}')
    else:
        st.sidebar.info(t("tms_prompt"))

    ref_H = st.session_state["tms_ref_H"]
    ref_C = st.session_state["tms_ref_C"]

elif shift_mode == t("shift_linear"):
    slope_H = st.sidebar.number_input(t("slope_h"), value=1.0)
    intercept_H = st.sidebar.number_input(t("intercept_h"), value=31.5)
    slope_C = st.sidebar.number_input(t("slope_c"), value=1.0)
    intercept_C = st.sidebar.number_input(t("intercept_c"), value=185.0)

element_filter = st.sidebar.selectbox(
    t("element_filter"),
    [t("all"), t("h"), t("c"), t("other")],
    index=0,
)

st.subheader(t("upload_header"))

opt_files = st.file_uploader(
    t("upload_opt"),
    type=["log", "out"],
    accept_multiple_files=True,
    key="opt_files",
)

giao_files = st.file_uploader(
    t("upload_giao"),
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
    st.caption(t("filename_prefix_info"))

st.subheader(t("settings_io_header"))
st.caption(t("save_project_desc"))
col_set1, col_set2, col_set3 = st.columns([2, 2, 1])

with col_set1:
    st.download_button(
        label=t("download_settings"),
        data=make_settings_json_bytes(),
        file_name=f"{output_prefix}_nmr_mapping_settings.json",
        mime="application/json",
    )

with col_set2:
    settings_file = st.file_uploader(
        t("upload_settings"),
        type=["json"],
        accept_multiple_files=False,
        key="settings_json",
    )
    if settings_file is not None and not st.session_state["settings_loaded_once"]:
        try:
            mappings_loaded = load_settings_json(settings_file)
            st.session_state["atom_mappings"] = mappings_loaded
            st.session_state["settings_loaded_once"] = True
            st.success(t("settings_loaded"))
            st.rerun()
        except Exception:
            st.error(t("settings_load_error"))
    if settings_file is None:
        st.session_state["settings_loaded_once"] = False

with col_set3:
    if st.button(t("clear_mappings")):
        st.session_state["atom_mappings"] = []
        st.session_state["mapping_selection"] = []
        st.rerun()

if opt_files and giao_files:
    opt_records = []
    for f in opt_files:
        text = read_text(f)
        cid = extract_conf_id(f.name)
        gibbs = extract_gibbs_free_energy(text)
        scf = extract_last_scf_energy(text)
        normal = check_normal_termination(text)
        xyz_text = extract_last_xyz_from_gaussian(text)

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

    if energy_mode == t("energy_gibbs"):
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
        st.error(t("no_valid"))
        st.stop()

    tms_ready = True
    if shift_mode == t("shift_tms") and (ref_H is None or ref_C is None):
        tms_ready = False
        st.warning(t("tms_not_ready"))

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
    st.session_state["latest_atom_table"] = atom_table_full.copy()

    valid_xyz = ""
    for _, row in valid_df.iterrows():
        if isinstance(row.get("xyz_text"), str) and row.get("xyz_text").strip():
            valid_xyz = row["xyz_text"]
            break
        if isinstance(row.get("giao_xyz_text"), str) and row.get("giao_xyz_text").strip():
            valid_xyz = row["giao_xyz_text"]
            break
    st.session_state["latest_xyz"] = valid_xyz

    per_conf_df = per_conf_df_full.copy()
    if element_filter == t("h"):
        per_conf_df = per_conf_df[per_conf_df["element"] == "H"].copy()
    elif element_filter == t("c"):
        per_conf_df = per_conf_df[per_conf_df["element"] == "C"].copy()
    elif element_filter == t("other"):
        per_conf_df = per_conf_df[~per_conf_df["element"].isin(["H", "C"])].copy()

    avg_df = add_boltzmann_average(per_conf_df, conf_ids, weights)

    if shift_mode == t("shift_manual"):
        result_df = shielding_to_shift(
            avg_df,
            mode="manual_reference",
            ref_H=ref_H,
            ref_C=ref_C,
        )
    elif shift_mode == t("shift_tms"):
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

    with st.expander(t("matched_header"), expanded=False):
        st.dataframe(pair_df, use_container_width=True)

    with st.expander(t("weights_header"), expanded=False):
        st.dataframe(valid_df, use_container_width=True)

    with st.expander(t("shielding_header"), expanded=False):
        st.dataframe(per_conf_df, use_container_width=True)

    with st.expander(t("avg_header"), expanded=False):
        st.caption(t("summary_columns_note"))
        st.dataframe(result_df, use_container_width=True)

st.subheader(t("mapping_header"))
st.write(t("mapping_desc"))

atom_df_ui = st.session_state["latest_atom_table"].copy()

if atom_df_ui.empty:
    st.info(t("coord_not_found"))
else:
    current_selection = st.session_state.get("mapping_selection", [])

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown(t("click_atoms"))
        picked_atoms = atom_picker(
            atom_df_ui["element"].tolist(),
            st.session_state["latest_xyz"],
            selected_atoms=current_selection,
            language=current_language(),
            key="nmr_atom_picker",
        )
        st.session_state["mapping_selection"] = picked_atoms
        current_selection = picked_atoms
        st.caption(t("viewer_caption"))

    with right:
        st.markdown(t("new_mapping"))

        if current_selection:
            selected_details = [
                f'{atom_df_ui.loc[atom_df_ui["atom_index"] == number, "element"].iloc[0]} {number}'
                for number in current_selection
                if number in atom_df_ui["atom_index"].tolist()
            ]
            st.success(t("selected", items=", ".join(selected_details)))
        else:
            st.info(t("none_selected"))

        label_mode = st.radio(
            t("label_entry"),
            [t("build_label"), t("free_text")],
            horizontal=True,
            key="nmr_label_mode",
        )

        if label_mode == t("build_label"):
            label_col1, label_col2 = st.columns([1.2, 1.0])
            position = label_col1.text_input(
                t("position_number"),
                value="",
                placeholder=t("position_placeholder"),
                key="nmr_position",
            ).strip()
            prime = label_col2.selectbox(
                t("prime"),
                ["", "′", "″", "‴"],
                key="nmr_prime",
            )
            proposed_label = f"H-{position}{prime}" if position else ""
            if proposed_label:
                st.markdown(t("display_label", label=proposed_label))
        else:
            proposed_label = st.text_input(
                t("label"),
                value="",
                placeholder=t("label_placeholder"),
                key="nmr_custom_label",
            ).strip()

        button_col1, button_col2 = st.columns(2)
        save_mapping = button_col1.button(
            t("save_mapping"),
            type="primary",
            use_container_width=True,
            key="nmr_save_mapping",
        )
        clear_selection = button_col2.button(
            t("clear_selection"),
            use_container_width=True,
            key="nmr_clear_selection",
        )

        if clear_selection:
            st.session_state["mapping_selection"] = []
            st.rerun()

        if save_mapping:
            if not proposed_label:
                st.error(t("enter_label"))
            elif not current_selection:
                st.error(t("select_atom"))
            else:
                non_h = [
                    n for n in current_selection
                    if atom_df_ui.loc[atom_df_ui["atom_index"] == n, "element"].iloc[0] != "H"
                ]
                if non_h:
                    st.error(t("non_hydrogen", atoms=", ".join(map(str, non_h))))
                else:
                    atom_indices = [atom_index_from_user_number(n) for n in current_selection]
                    existing = next(
                        (item for item in st.session_state["atom_mappings"] if item["label"] == proposed_label),
                        None,
                    )
                    if existing is None:
                        st.session_state["atom_mappings"].append(
                            {
                                "label": proposed_label,
                                "atom_numbers": list(current_selection),
                                "atom_indices": atom_indices,
                            }
                        )
                        st.success(t("added", label=proposed_label))
                    else:
                        existing["atom_numbers"] = list(current_selection)
                        existing["atom_indices"] = atom_indices
                        st.success(t("updated", label=proposed_label))
                    st.session_state["mapping_selection"] = []
                    st.rerun()

        with st.expander(t("manual_fallback")):
            st.caption(t("manual_desc"))
            manual_raw = st.text_input(
                t("manual_numbers"),
                value=",".join(map(str, current_selection)),
                key="nmr_manual_selection",
            )
            if st.button(t("apply_manual"), key="nmr_apply_manual"):
                try:
                    manual_numbers = sorted({
                        int(x.strip())
                        for x in manual_raw.split(",")
                        if x.strip()
                    })
                    invalid = [n for n in manual_numbers if n < 1 or n > len(atom_df_ui)]
                    if invalid:
                        st.error(t("out_of_range", atoms=", ".join(map(str, invalid))))
                    else:
                        st.session_state["mapping_selection"] = manual_numbers
                        st.rerun()
                except ValueError:
                    st.error(t("integer_error"))

    st.markdown(t("registered_mappings"))
    if not st.session_state["atom_mappings"]:
        st.info(t("no_mappings"))
    else:
        for idx, item in enumerate(list(st.session_state["atom_mappings"])):
            with st.container(border=True):
                info_col, select_col, delete_col = st.columns([4.5, 1.5, 1.0])
                atom_numbers = [int(n) for n in item.get("atom_numbers", [])]
                atom_text = ", ".join(str(n) for n in atom_numbers) if atom_numbers else t("not_assigned")
                info_col.markdown(f'**{item["label"]}**  \n{t("atoms_label")}: {atom_text}')
                if select_col.button(
                    t("show_edit"),
                    key=f"show_mapping_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["mapping_selection"] = list(item["atom_numbers"])
                    st.rerun()
                if delete_col.button(
                    t("delete"),
                    key=f"delete_mapping_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["atom_mappings"].pop(idx)
                    st.rerun()

    with st.expander(t("mapping_status"), expanded=False):
        status_df = pd.DataFrame([
            {
                t("mapping_registered"): len(st.session_state["atom_mappings"]),
                t("mapping_labels"): ", ".join(item["label"] for item in st.session_state["atom_mappings"]),
            }
        ])
        st.dataframe(status_df, use_container_width=True, hide_index=True)

if result_df is not None:
    if st.session_state["atom_mappings"]:
        eq_df = average_equivalent_atoms(result_df, st.session_state["atom_mappings"])
        st.subheader(t("eq_avg_header"))
        st.dataframe(eq_df, use_container_width=True)

    st.subheader(t("download_header"))

    if per_conf_df is not None:
        st.download_button(
            label=t("download_per_conf"),
            data=dataframe_to_csv_bytes(per_conf_df),
            file_name=f"{output_prefix}_per_conformer_isotropic_shieldings.csv",
            mime="text/csv",
        )

    st.download_button(
        label=t("download_avg"),
        data=dataframe_to_csv_bytes(result_df),
        file_name=f"{output_prefix}_boltzmann_averaged_nmr.csv",
        mime="text/csv",
    )

    if valid_df is not None:
        st.download_button(
            label=t("download_weights"),
            data=dataframe_to_csv_bytes(valid_df),
            file_name=f"{output_prefix}_boltzmann_weights.csv",
            mime="text/csv",
        )

    if eq_df is not None:
        st.download_button(
            label=t("download_eq"),
            data=dataframe_to_csv_bytes(eq_df),
            file_name=f"{output_prefix}_equivalent_atom_averaged_nmr.csv",
            mime="text/csv",
        )
