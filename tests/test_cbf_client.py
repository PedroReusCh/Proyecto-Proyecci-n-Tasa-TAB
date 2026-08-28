import io
import pandas as pd
import pytest
from src.cbf_client import (
    parse_fixing_rates_excel,
    clean_tab_uf_dataframe,
    get_clean_tab_uf_dataset,
    CBFDataResult,
)

def test_parse_fixing_rates_excel_structure():
    # Mock con formato exacto de CBF
    data = [
        [None] * 15,
        [None] * 15,
        [None, None, "Date", None, None, None, None, "TAB UF\n(percentage, annual basis)", None, None] + [None] * 5,
        [None, None, "Tenor", "30 Days", "90 Days", "180 Days", "360 Days", "90 Days", "180 Days", "360 Days"] + [None] * 5,
        [None] * 15,
        [None, None, "2024-01-02", 0.5, 1.0, 1.5, 2.0, 2.75, 2.90, 3.10] + [None] * 5,
        [None, None, "2024-01-03", 0.5, 1.0, 1.5, 2.0, 2.78, 2.92, 3.12] + [None] * 5,
        [None, None, "2024-01-04", 0.5, 1.0, 1.5, 2.0, 2.80, 2.95, 3.15] + [None] * 5,
    ]
    df_raw = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="FixingRates", index=False, header=False)
    
    parsed_df = parse_fixing_rates_excel(buf.getvalue())
    assert not parsed_df.empty
    assert list(parsed_df.columns) == ["Date", "TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]
    assert len(parsed_df) == 3
    assert parsed_df.iloc[0]["TAB_UF_90"] == 2.75
    assert parsed_df.iloc[0]["TAB_UF_180"] == 2.90
    assert parsed_df.iloc[0]["TAB_UF_360"] == 3.10

def test_clean_tab_uf_dataframe_sorts_and_cleans():
    raw_data = {
        "Date": ["2024-01-05", "2024-01-02", "invalida", "2024-01-03"],
        "TAB_UF_90": [2.85, 2.75, 2.0, "error"],
        "TAB_UF_180": [2.95, 2.90, 2.1, 2.92],
        "TAB_UF_360": [3.20, 3.10, 2.5, 3.12],
    }
    df = pd.DataFrame(raw_data)
    cleaned = clean_tab_uf_dataframe(df)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["Date"] == pd.Timestamp("2024-01-02")
    assert cleaned.iloc[1]["Date"] == pd.Timestamp("2024-01-05")

def test_get_clean_tab_uf_dataset_returns_result():
    data = [
        [None] * 15,
        [None] * 15,
        [None, None, "Date", None, None, None, None, "TAB UF", None, None] + [None] * 5,
        [None, None, "Tenor", "30 Days", "90 Days", "180 Days", "360 Days", "90 Days", "180 Days", "360 Days"] + [None] * 5,
        [None] * 15,
        [None, None, "2024-01-02", 0.5, 1.0, 1.5, 2.0, 2.75, 2.90, 3.10] + [None] * 5,
    ]
    df_raw = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="FixingRates", index=False, header=False)
    
    result = get_clean_tab_uf_dataset(buf.getvalue())
    assert isinstance(result, CBFDataResult)
    assert result.records_count == 1
    assert result.last_date == pd.Timestamp("2024-01-02")
    assert result.latest_rates["TAB_UF_90"] == 2.75
    assert result.latest_rates["TAB_UF_180"] == 2.90
    assert result.latest_rates["TAB_UF_360"] == 3.10
