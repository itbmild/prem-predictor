""" Hanldes saving and loading logic for evaluation reports and figures """
from pathlib import Path
from matplotlib.figure import Figure
import json

class EvaluationStore:
    def __init__(self, reports_dir: str):
        self.reports_dir = Path(reports_dir)

    def save_report(self, model_id: str, evaluation: dict) -> str:
        """ Saves evaluation report as JSON for a model """
        report_path = self.reports_dir / model_id
        report_path.mkdir(parents=True, exist_ok=False)

        # save the evaluation dictionary in JSON format
        with open(report_path / f"{model_id}_test.json", "w") as f:
            json.dump(evaluation, f, indent=2)

        return str(report_path)
    
    def load_report(self, model_id: str) -> dict:
        """ Load evaluation report for model corresponding to model_id """
        report_path = self.reports_dir / model_id

        if not report_path.exists():
            raise FileNotFoundError(f"No evaluation report exists for model with ID: {model_id}")
        
        with open(report_path, "r") as f:
            return json.load(f)
        
    def save_figure(self, model_id: str, figure_name:str, figure: Figure) -> str:
        """ Save a matplotlib figure from a specific model evaluation """
        figure_path = self.reports_dir / model_id / "figures" / figure_name
        figure.savefig(figure_path)
        return str(figure_path)
    




        
    

