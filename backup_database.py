import zipfile
from datetime import datetime
from pathlib import Path

# files to backup
files_to_backup = [
    r"C:\Users\tmerri03\Desktop\Temp Neural Files\neuron_labels.json",
    r"C:\Users\tmerri03\Desktop\Temp Neural Files\neurons.db",
    r"C:\Users\tmerri03\Desktop\Temp Neural Files\HVC Response Strengths.json",
]

backup_temp_folder = Path(r"C:\Users\tmerri03\Desktop\Temp Neural Files")

# timestamped zip name
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
zip_path = backup_temp_folder / f"neural_backup_{date_str}.zip"

print("Creating zip:", zip_path)

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for file in files_to_backup:
        file_path = Path(file)
        z.write(file_path, arcname=file_path.name)

print("Zip created!")

import subprocess

rclone_exe = r'C:\Users\tmerri03\Documents\rclone\rclone.exe'
dest = r"Rstore:/as_rsch_kao_lab01$/Data/tyler/Recordings/Backups/daily_snapshots"

cmd = [
    rclone_exe,
    'copy',
    str(zip_path),
    dest
]

print("Uploading zip:", " ".join(cmd))
subprocess.run(cmd, check=True)

zip_path.unlink()
print("Local zip removed")
