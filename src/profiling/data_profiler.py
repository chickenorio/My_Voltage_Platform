import pandas as pd
from pathlib import Path
from typing import Union, Optional
import logging
from ydata_profiling import ProfileReport
import sweetviz as sv
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProfiler:
    def __init__(
        self,
        output_dir: Union[str, Path] = "../data/processed/reports",
        tool: str = "ydata"
    ):
        """
        Initialize DataProfiler with output directory and preferred tool.
        
        Args:
            output_dir: Directory to save reports
            tool: Profiling tool to use ('ydata' or 'sweetviz')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool = tool.lower()
        
    def generate_report(
        self,
        df: pd.DataFrame,
        title: str = "Data Quality Report",
        minimal: bool = False
    ) -> Path:
        """
        Generate a data profiling report using the selected tool.
        
        Args:
            df: DataFrame to profile
            title: Report title
            minimal: Whether to generate a minimal report
        
        Returns:
            Path: Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.tool == "ydata":
            return self._generate_ydata_report(df, title, timestamp, minimal)
        elif self.tool == "sweetviz":
            return self._generate_sweetviz_report(df, title, timestamp)
        else:
            raise ValueError(f"Unsupported profiling tool: {self.tool}")
    
    def _generate_ydata_report(
        self,
        df: pd.DataFrame,
        title: str,
        timestamp: str,
        minimal: bool
    ) -> Path:
        """Generate report using ydata-profiling."""
        logger.info("Generating ydata-profiling report...")
        
        profile = ProfileReport(
            df,
            title=title,
            minimal=minimal,
            explorative=not minimal,
            dark_mode=True
        )
        
        output_path = self.output_dir / f"ydata_report_{timestamp}.html"
        profile.to_file(output_path)
        
        # Save summary statistics
        stats = {
            "timestamp": timestamp,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": df.isna().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        stats_path = self.output_dir / f"profile_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def _generate_sweetviz_report(
        self,
        df: pd.DataFrame,
        title: str,
        timestamp: str
    ) -> Path:
        """Generate report using sweetviz."""
        logger.info("Generating sweetviz report...")
        
        report = sv.analyze(df)
        output_path = self.output_dir / f"sweetviz_report_{timestamp}.html"
        report.show_html(str(output_path))
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def compare_datasets(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        labels: tuple = ("Original", "Modified")
    ) -> Path:
        """
        Generate a comparison report between two datasets.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            labels: Labels for the two datasets
        
        Returns:
            Path: Path to the comparison report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.tool == "sweetviz":
            report = sv.compare(df1, df2, labels[0], labels[1])
            output_path = self.output_dir / f"comparison_report_{timestamp}.html"
            report.show_html(str(output_path))
        else:
            # ydata-profiling comparison
            report1 = ProfileReport(df1, title=f"{labels[0]} Dataset")
            report2 = ProfileReport(df2, title=f"{labels[1]} Dataset")
            comparison = report1.compare(report2)
            
            output_path = self.output_dir / f"comparison_report_{timestamp}.html"
            comparison.to_file(output_path)
        
        logger.info(f"Comparison report saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    try:
        from ..data_ingest.data_loader import load_csv
        
        df = load_csv("../data/raw/sample.csv")
        profiler = DataProfiler()
        
        # Generate individual report
        report_path = profiler.generate_report(df, "Sample Data Profile")
        print(f"Report generated at: {report_path}")
        
        # Generate comparison report with modified data
        df_modified = df.copy()
        df_modified['voltage'] = df_modified['voltage'].fillna(df_modified['voltage'].mean())
        comparison_path = profiler.compare_datasets(df, df_modified)
        print(f"Comparison report generated at: {comparison_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate reports: {str(e)}") 