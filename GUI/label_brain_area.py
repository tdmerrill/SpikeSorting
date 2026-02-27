import streamlit as st
import json
from pathlib import Path
import pandas as pd
import os

st.set_page_config(layout="wide")
st.title("Neuron Brain Area Labeler")

# -------- LOAD INPUT DATA --------
with open("neurons_to_label.json") as f:
    birds = json.load(f)

areas_file = Path(r"C:\Users\tmerri03\PycharmProjects\SpikeSorting\GUI\brain_areas.json")
if areas_file.exists():
    AREAS = json.load(open(areas_file))["areas"]
else:
    AREAS = ["", "HVC", "RA", "LMAN"]

base_dir = r'C:\Users\tmerri03\Desktop\Temp Neural Files'
labels_path = Path(os.path.join(base_dir, "neuron_labels.json"))
if labels_path.exists():
    labels = json.load(open(labels_path))
else:
    labels = {}

# ---------- SESSION STATE ----------
if "labels_state" not in st.session_state:
    st.session_state.labels_state = labels

# helper
def get_label(bird, rec, neuron):
    return st.session_state.labels_state.get(bird, {}).get(rec, {}).get(str(neuron), "")

def set_label(bird, rec, neuron, area):
    st.session_state.labels_state.setdefault(bird, {}).setdefault(rec, {})[str(neuron)] = area

bird_tabs = st.tabs(list(birds.keys()))

# -------- UI --------
for bird_tab, bird in zip(bird_tabs, birds.keys()):
    with bird_tab:
        st.header(f"Bird: {bird}")
        rec_tabs = st.tabs(list(birds[bird].keys()))

        for rec_tab, recording in zip(rec_tabs, birds[bird].keys()):
            with rec_tab:
                st.subheader(f"Recording: {recording}")
                neurons = birds[bird][recording]

                rows = []
                for neuron in sorted(neurons):
                    rows.append({
                        "select": False,
                        "neuron_id": neuron,
                        "brain_area": get_label(bird, recording, neuron)
                    })

                df = pd.DataFrame(rows)

                edited = st.data_editor(
                    df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                        "brain_area": st.column_config.SelectboxColumn(
                            "Brain Area",
                            options=AREAS
                        )
                    },
                    key=f"{bird}_{recording}_editor"
                )

                # -------- BULK ASSIGN CONTROLS --------
                col1, col2 = st.columns([3, 1])

                with col1:
                    bulk_area = st.selectbox(
                        "Set selected neurons to:",
                        AREAS,
                        key=f"{bird}_{recording}_bulk_area"
                    )

                with col2:
                    apply_clicked = st.button(
                        "Apply",
                        key=f"{bird}_{recording}_apply"
                    )

                if apply_clicked:
                    for _, row in edited.iterrows():
                        if row["select"]:
                            set_label(bird, recording, row.neuron_id, bulk_area)

                    st.success("Updated selected neurons.")
                    st.rerun()

                # -------- SAVE EDITS BACK --------
                for _, row in edited.iterrows():
                    set_label(bird, recording, row.neuron_id, row.brain_area)

# -------- AUTO SAVE --------
def autosave():
    with open(labels_path, "w") as f:
        json.dump(st.session_state.labels_state, f, indent=2)

autosave()
st.caption("Autosave active")

# -------- SAVE BUTTON --------
if st.button("💾 Save labels"):
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    st.success("Labels saved to neuron_labels.json")
