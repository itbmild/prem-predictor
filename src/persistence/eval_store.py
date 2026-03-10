""" Hanldes saving and loading logic for evaluation reports and figures """
from pathlib import Path
from matplotlib.figure import Figure
import json

class EvaluationStore:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)

    def save_evaluation(self, model_id: str, evaluation: dict):
        """ Saves evaluation given evaluation output from model
         
        Evaluation dict:
            {
              "report": dict[str, int]
              "figures": dict[str, Figure]
            }  
           
        """
        artifact_dir = self.artifacts_dir / model_id
        eval_path = artifact_dir / "evaluation"
        eval_path.mkdir(parents=True, exist_ok=True)

        self._save_report(eval_path, evaluation["report"])
        self._save_figures(eval_path, evaluation["figures"])


    def _save_report(eval_path: Path, report: dict):
        """ Helper for saving report dict to specified path """
        with open(eval_path / "eval_report", "w") as f:
            json.dump(report, f, indent=2)

    def _save_figures(eval_path: Path, figures: dict[str, Figure]):
        """ Saves list of provided figures in specified directory """
        fig_dir = eval_path / "figures"
        for name, fig in figures.items():
            path = fig_dir / f"{name}.png"
            fig.savefig(path)
    
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
    




        
    

