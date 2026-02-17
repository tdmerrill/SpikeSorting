import sqlite3
import os

DB_PATH = r"C:\Users\tmerri03\Desktop\Temp Neural Files\neurons.db"

def connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = connect()
    cur = conn.cursor()

    # One row per neuron (logical unit)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS neurons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        
        --Identity
        session_id TEXT,
        probe TEXT,
        unit_id INTEGER,
        
        --Unit information
        snr REAL,
        firing_rate REAL,
        isi_violation_ratio REAL,
        presence_ratio REAL,
        sliding_rp_violation REAL,
        drift REAL,
        amplitude_median REAL,
        amplitude_cv REAL,
        noise_cutoff REAL,
        spike_width_pp REAL,
        spike_width_hw REAL,
        unit_loc_x REAL,
        unit_loc_y REAL,
        
        --Paths to external data
        spike_file TEXT,
        stimulus_file TEXT,
        
        UNIQUE(session_id, probe, unit_id)
    )
    """)

    conn.commit()
    conn.close()
