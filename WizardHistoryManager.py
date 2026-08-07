import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any

WIZARD_HISTORY_FILE = "wizard_history.json"

class WizardHistoryManager:
    def __init__(self, filepath=WIZARD_HISTORY_FILE):
        self.filepath = filepath
        self.data: Dict[str, Any] = {"date": None, "recommendations": []}
        self.load()
        self.check_reset()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading wizard history: {e}")
                self.data = {"date": None, "recommendations": []}

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=4)

    def check_reset(self):
        """Reset the history if the current date is different from the saved date.
           Technically, if it's before 09:15, it might still be 'today' in calendar terms,
           but resetting at midnight is sufficient for a 'daily' timeline.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self.data.get("date") != today_str:
            self.data = {"date": today_str, "recommendations": []}
            self.save()

    def add_recommendation(self, strategy_name, legs, rationale, score):
        self.check_reset()
        rec = {
            "id": str(uuid.uuid4()),
            "time": datetime.now().strftime("%H:%M:%S"),
            "strategy": strategy_name,
            "legs": legs,
            "rationale": rationale,
            "score": score
        }
        self.data["recommendations"].append(rec)
        self.save()
        return rec

    def get_today_history(self):
        self.check_reset()
        return self.data["recommendations"]
