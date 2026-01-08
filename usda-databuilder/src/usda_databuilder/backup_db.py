from datetime import datetime
from pathlib import Path
import shutil

def backup():
    DB = Path("data/usda.db")
    BACKUP_DIR = Path("data/backups")
    BACKUP_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_path = BACKUP_DIR / f"usda_{timestamp}.db"

    shutil.copy(DB, backup_path)
    print(f"✔ Backup créé : {backup_path}")
