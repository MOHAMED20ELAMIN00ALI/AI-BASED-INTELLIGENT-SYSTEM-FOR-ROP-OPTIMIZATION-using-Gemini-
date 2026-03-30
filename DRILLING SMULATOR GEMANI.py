import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import json
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCUv1Fwn8bexJ7NLOpl3wUuGFxpDKK9oDE"

genai.configure(api_key=GEMINI_API_KEY)

# ═══════════════════════════════════════════════════════════
# Logging Setup - Enhanced
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DrillingApp")

try:
    file_handler = logging.FileHandler("drilling_app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s'
    ))
    logger.addHandler(file_handler)
    logger.info("=" * 60)
    logger.info("Drilling Application Started")
    logger.info("=" * 60)
except Exception as e:
    logger.warning(f"Could not create log file: {e}")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

class ModelConstants:
    M_STAR_INTERCEPT = 1359.1
    M_STAR_SLOPE = 714.19
    W_STAR_FACTOR = 7.875
    R_CUBIC_COEF = 0.00004348
    A_QUAD_COEF = 0.928125
    A_LINEAR_COEF = 6.0
    A_CONSTANT = 1.0
    DEFAULT_BIT_DIAMETER = 12.25
    WOB_MIN, WOB_MAX = 10000, 80000
    RPM_MIN, RPM_MAX = 40, 250


COLORS = {
    "primary": "#3b82f6", "secondary": "#1d4ed8", "accent": "#0ea5e9",
    "success": "#22c55e", "warning": "#f59e0b", "danger": "#ef4444",
    "light": "#f8fafc", "lighter": "#ffffff", "card": "#ffffff",
    "card_hover": "#f1f5f9", "text": "#1e293b", "text_dim": "#64748b",
    "border": "#e2e8f0", "chart_bg": "#ffffff", "grid_color": "#e2e8f0",
    "monte_carlo_mean": "#06b6d4", "monte_carlo_ci": "#8b5cf6",
    "monte_carlo_scatter": "#f97316",
    "dfdT_color": "#3b82f6",
    "dfdD_color": "#ef4444",
    "wob_color": "#3b82f6",
    "rpm_color": "#8b5cf6",
    "time_color": "#22c55e",
    "safe": "#22c55e",
    "caution": "#f59e0b",
    "unsafe": "#ef4444",
    "optimal": "#8b5cf6"
}

TABLE_1_6 = {
    "1/8": {"U": 123, "Z_1.0": 89, "Z_0.75": 97.0, "Z_0.5": 105, "Z_0": 123},
    "2/8": {"U": 316, "Z_1.0": 179, "Z_0.75": 207.5, "Z_0.5": 236, "Z_0": 316},
    "3/8": {"U": 581, "Z_1.0": 268, "Z_0.75": 328.5, "Z_0.5": 389, "Z_0": 581},
    "4/8": {"U": 920, "Z_1.0": 357, "Z_0.75": 460.0, "Z_0.5": 563, "Z_0": 920},
    "5/8": {"U": 1337, "Z_1.0": 446, "Z_0.75": 601.0, "Z_0.5": 756, "Z_0": 1337},
    "6/8": {"U": 1834, "Z_1.0": 536, "Z_0.75": 751.5, "Z_0.5": 967, "Z_0": 1834},
    "7/8": {"U": 2413, "Z_1.0": 625, "Z_0.75": 909.5, "Z_0.5": 1194, "Z_0": 2413},
    "8/8": {"U": 3078, "Z_1.0": 714, "Z_0.75": 1075.5, "Z_0.5": 1437, "Z_0": 3078}
}


# ═══════════════════════════════════════════════════════════
# Gemini AI Analyzer (unchanged math/AI logic)
# ═══════════════════════════════════════════════════════════

class GeminiDrillingAnalyzer:

    def __init__(self):
        self.model = None
        self.is_configured = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
                self.model = genai.GenerativeModel('gemini-3-flash-preview')
                self.is_configured = True
                logger.info("Gemini AI initialized successfully")
            else:
                logger.warning("Gemini API key not configured")
                self.is_configured = False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.is_configured = False

    def analyze_drilling_data(self, results) -> Dict:
        if not self.is_configured:
            raise Exception("Gemini API not configured!")
        if results is None or len(results.dfdT_all) == 0:
            raise Exception("No data to analyze!")

        logger.info(f"AI analysis started on {len(results.dfdT_all)} data points")
        n_points = len(results.dfdT_all)

        stats = {
            'dfdT_mean': float(np.mean(results.dfdT_all)),
            'dfdT_std': float(np.std(results.dfdT_all)),
            'dfdT_min': float(np.min(results.dfdT_all)),
            'dfdT_max': float(np.max(results.dfdT_all)),
            'dfdD_mean': float(np.mean(results.dfdD_all)),
            'dfdD_std': float(np.std(results.dfdD_all)),
            'dfdD_min': float(np.min(results.dfdD_all)),
            'dfdD_max': float(np.max(results.dfdD_all)),
            'wob_mean': float(np.mean(results.wob_valid)),
            'wob_std': float(np.std(results.wob_valid)),
            'wob_min': float(np.min(results.wob_valid)),
            'wob_max': float(np.max(results.wob_valid)),
            'rpm_mean': float(np.mean(results.rpm_valid)),
            'rpm_std': float(np.std(results.rpm_valid)),
            'rpm_min': float(np.min(results.rpm_valid)),
            'rpm_max': float(np.max(results.rpm_valid)),
            'n_points': n_points
        }

        try:
            ai_analysis = self._get_gemini_analysis(stats, results)
            if ai_analysis:
                logger.info("AI analysis completed successfully")
                return ai_analysis
            else:
                raise Exception("Gemini returned no results")
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise Exception(f"Gemini analysis failed: {str(e)}")

    def _get_gemini_analysis(self, stats: Dict, results) -> Optional[Dict]:
        prompt = f"""
        You are an expert drilling engineer. Analyze this drilling simulation data and provide safety classification thresholds.

        Statistical Summary:
        - dF/dT (Rate of Penetration): mean={stats['dfdT_mean']:.2f}, std={stats['dfdT_std']:.2f}, min={stats['dfdT_min']:.2f}, max={stats['dfdT_max']:.2f}
        - dF/dD (Wear Rate): mean={stats['dfdD_mean']:.4f}, std={stats['dfdD_std']:.4f}, min={stats['dfdD_min']:.4f}, max={stats['dfdD_max']:.4f}
        - WOB: mean={stats['wob_mean']:.0f} lbs, std={stats['wob_std']:.0f}, range=[{stats['wob_min']:.0f}, {stats['wob_max']:.0f}]
        - RPM: mean={stats['rpm_mean']:.0f}, std={stats['rpm_std']:.0f}, range=[{stats['rpm_min']:.0f}, {stats['rpm_max']:.0f}]
        - Total simulation points: {stats['n_points']}

        Based on drilling engineering best practices and the Galle-Woods model:
        1. Define safe operating thresholds for dF/dT (rate of penetration)
        2. Define maximum acceptable dF/dD (wear rate)
        3. Determine optimal WOB and RPM ranges for safe operation
        4. Assess overall risk level

        IMPORTANT: Respond ONLY with this exact JSON format, no additional text:
        {{
            "safe_dfdT_range": [minimum_safe_dfdT, maximum_safe_dfdT],
            "safe_dfdD_max": maximum_acceptable_wear_rate,
            "optimal_wob_range": [min_optimal_wob, max_optimal_wob],
            "optimal_rpm_range": [min_optimal_rpm, max_optimal_rpm],
            "risk_level": "low" or "medium" or "high",
            "recommendation": "Brief engineering recommendation in English"
        }}
        """

        try:
            logger.debug("Sending prompt to Gemini API")
            response = self.model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Gemini response received: {len(response_text)} chars")

            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                ai_result = json.loads(json_str)
                logger.info("Gemini JSON parsed successfully")
                return self._apply_gemini_classification(results, ai_result, stats)
            else:
                raise Exception("Could not parse JSON from Gemini response")
        except Exception as e:
            logger.error(f"Gemini response parsing error: {e}")
            raise

    def _apply_gemini_classification(self, results, ai_result: Dict, stats: Dict) -> Dict:
        n_points = len(results.dfdT_all)
        safety_colors = np.empty(n_points, dtype='U10')

        safe_dfdT = ai_result.get('safe_dfdT_range', [stats['dfdT_mean'] * 0.5, stats['dfdT_mean'] * 1.5])
        safe_dfdD_max = ai_result.get('safe_dfdD_max', stats['dfdD_mean'] + 2 * stats['dfdD_std'])
        optimal_wob = ai_result.get('optimal_wob_range',
                                    [stats['wob_mean'] - stats['wob_std'], stats['wob_mean'] + stats['wob_std']])
        optimal_rpm = ai_result.get('optimal_rpm_range',
                                    [stats['rpm_mean'] - stats['rpm_std'], stats['rpm_mean'] + stats['rpm_std']])

        optimal_indices = []
        for i in range(n_points):
            dfdT = results.dfdT_all[i]
            dfdD = results.dfdD_all[i]
            wob = results.wob_valid[i]
            rpm = results.rpm_valid[i]

            is_safe_dfdT = safe_dfdT[0] <= dfdT <= safe_dfdT[1]
            is_safe_dfdD = dfdD <= safe_dfdD_max
            is_optimal_wob = optimal_wob[0] <= wob <= optimal_wob[1]
            is_optimal_rpm = optimal_rpm[0] <= rpm <= optimal_rpm[1]

            if is_safe_dfdT and is_safe_dfdD and is_optimal_wob and is_optimal_rpm:
                safety_colors[i] = 'optimal'
                optimal_indices.append(i)
            elif is_safe_dfdT and is_safe_dfdD:
                safety_colors[i] = 'safe'
            elif not is_safe_dfdD or dfdT > safe_dfdT[1] * 1.5:
                safety_colors[i] = 'unsafe'
            else:
                safety_colors[i] = 'caution'

        optimal_list = []
        best_options = {}

        if len(optimal_indices) > 0:
            for idx in optimal_indices:
                optimal_list.append({
                    'index': idx,
                    'wob': float(results.wob_valid[idx]),
                    'rpm': float(results.rpm_valid[idx]),
                    'dfdT': float(results.dfdT_all[idx]),
                    'dfdD': float(results.dfdD_all[idx])
                })
            sorted_optimal = sorted(optimal_list, key=lambda x: (-x['dfdT'], -x['dfdD']))
            top_1 = sorted_optimal[0]
            top_2 = sorted_optimal[1] if len(sorted_optimal) > 1 else None
            top_3 = sorted_optimal[2] if len(sorted_optimal) > 2 else None
            best_options = {"option_1": top_1, "option_2": top_2, "option_3": top_3}
            best_idx = top_1['index']
        else:
            safe_indices = np.where(safety_colors == 'safe')[0]
            if len(safe_indices) > 0:
                safe_list = []
                for idx in safe_indices:
                    safe_list.append({
                        'index': idx,
                        'wob': float(results.wob_valid[idx]),
                        'rpm': float(results.rpm_valid[idx]),
                        'dfdT': float(results.dfdT_all[idx]),
                        'dfdD': float(results.dfdD_all[idx])
                    })
                sorted_safe = sorted(safe_list, key=lambda x: (-x['dfdT'], -x['dfdD']))
                best_idx = sorted_safe[0]['index']
                best_options = {
                    "option_1": sorted_safe[0],
                    "option_2": sorted_safe[1] if len(sorted_safe) > 1 else None,
                    "option_3": sorted_safe[2] if len(sorted_safe) > 2 else None
                }
            else:
                all_points = []
                for i in range(n_points):
                    all_points.append({
                        'index': i,
                        'wob': float(results.wob_valid[i]),
                        'rpm': float(results.rpm_valid[i]),
                        'dfdT': float(results.dfdT_all[i]),
                        'dfdD': float(results.dfdD_all[i])
                    })
                sorted_all = sorted(all_points, key=lambda x: (-x['dfdT'], -x['dfdD']))
                best_idx = sorted_all[0]['index']
                best_options = {
                    "option_1": sorted_all[0],
                    "option_2": sorted_all[1] if len(sorted_all) > 1 else None,
                    "option_3": sorted_all[2] if len(sorted_all) > 2 else None
                }

        rejection_stats = {
            'unsafe_count': int(np.sum(safety_colors == 'unsafe')),
            'caution_count': int(np.sum(safety_colors == 'caution')),
            'safe_count': int(np.sum(safety_colors == 'safe')),
            'optimal_count': int(np.sum(safety_colors == 'optimal')),
            'reason_unsafe': f"Exceeded wear limit ({safe_dfdD_max:.4f}) or excessive speed",
            'reason_caution': "Unstable parameters or low energy efficiency"
        }

        logger.info(
            f"Classification: optimal={rejection_stats['optimal_count']}, "
            f"safe={rejection_stats['safe_count']}, "
            f"caution={rejection_stats['caution_count']}, "
            f"unsafe={rejection_stats['unsafe_count']}"
        )

        return {
            'safety_colors': safety_colors,
            'optimal_indices': optimal_indices,
            'best_point_index': best_idx,
            'best_point': {
                'wob': float(results.wob_valid[best_idx]),
                'rpm': float(results.rpm_valid[best_idx]),
                'dfdT': float(results.dfdT_all[best_idx]),
                'dfdD': float(results.dfdD_all[best_idx])
            },
            'top_3_options': best_options,
            'rejection_analysis': rejection_stats,
            'thresholds': {
                'safe_dfdT_range': [float(f) for f in safe_dfdT],
                'safe_dfdD_max': float(safe_dfdD_max),
                'optimal_wob_range': [float(f) for f in optimal_wob],
                'optimal_rpm_range': [float(f) for f in optimal_rpm]
            },
            'ai_recommendation': ai_result.get('recommendation', 'Analysis complete.'),
            'risk_level': ai_result.get('risk_level', 'unknown'),
            'analysis_source': 'gemini',
            'selection_criteria': 'Highest dF/dT first, then Highest dF/dD'
        }


# ═══════════════════════════════════════════════════════════
# Data Cleaner - Enhanced with report
# ═══════════════════════════════════════════════════════════

class DataCleaner:
    def __init__(self):
        self.report = {}
        logger.info("DataCleaner initialized")

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        logger.info(f"Starting data cleaning: {len(df)} rows, {len(df.columns)} columns")

        report = {
            'original_rows': len(df),
            'original_columns': list(df.columns),
            'removed_rows': 0,
            'issues_found': [],
            'columns_cleaned': [],
            'outliers_removed': 0,
            'missing_filled': 0,
            'duplicates_removed': 0,
            'empty_rows_removed': 0,
            'na_rows_removed': 0,
            'final_rows': 0,
            'success': False,
            'column_stats_before': {},
            'column_stats_after': {},
        }

        numeric_columns = ['W', 'N', 'F', 'T', 'H', 'D']

        # Stats before cleaning
        for col in numeric_columns:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                report['column_stats_before'][col] = {
                    'count': int(vals.notna().sum()),
                    'missing': int(vals.isna().sum()),
                    'min': float(vals.min()) if vals.notna().any() else None,
                    'max': float(vals.max()) if vals.notna().any() else None,
                    'mean': float(vals.mean()) if vals.notna().any() else None,
                }
                logger.debug(f"Column '{col}' before: {report['column_stats_before'][col]}")

        # 1. Drop fully empty rows
        original_count = len(df)
        df = df.dropna(how='all')
        empty_rows = original_count - len(df)
        report['empty_rows_removed'] = empty_rows
        if empty_rows > 0:
            report['issues_found'].append(f"Removed {empty_rows} completely empty rows")
            logger.info(f"Removed {empty_rows} empty rows")

        # 2. Strip column names
        df.columns = df.columns.str.strip()

        # 3. Coerce numeric columns
        for col in numeric_columns:
            if col in df.columns:
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                after_na = df[col].isna().sum()
                coerced = after_na - before_na
                if coerced > 0:
                    report['issues_found'].append(
                        f"Column '{col}': {coerced} non-numeric values converted to NaN"
                    )
                    logger.warning(f"Column '{col}': {coerced} values coerced to NaN")
                report['columns_cleaned'].append(col)

        # 4. Drop duplicates
        before_dup = len(df)
        df = df.drop_duplicates()
        duplicates = before_dup - len(df)
        report['duplicates_removed'] = duplicates
        if duplicates > 0:
            report['issues_found'].append(f"Removed {duplicates} duplicate rows")
            logger.info(f"Removed {duplicates} duplicate rows")

        # 5. Drop rows with missing numeric values
        existing_numeric = [c for c in numeric_columns if c in df.columns]
        before_na = len(df)
        df = df.dropna(subset=existing_numeric)
        na_removed = before_na - len(df)
        report['na_rows_removed'] = na_removed
        if na_removed > 0:
            report['issues_found'].append(f"Removed {na_removed} rows with missing values")
            logger.info(f"Removed {na_removed} rows with missing values")

        # 6. Remove outliers (3x IQR)
        outlier_columns = ['W', 'N', 'F', 'T']
        total_outliers = 0
        for col in outlier_columns:
            if col in df.columns and len(df) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo = Q1 - 3 * IQR
                hi = Q3 + 3 * IQR
                outliers = (df[col] < lo) | (df[col] > hi)
                cnt = outliers.sum()
                if 0 < cnt < len(df) * 0.1:
                    df = df[~outliers]
                    total_outliers += cnt
                    report['issues_found'].append(
                        f"Column '{col}': removed {cnt} outliers ([{lo:.1f}, {hi:.1f}])"
                    )
                    logger.info(f"Column '{col}': removed {cnt} outliers")
        report['outliers_removed'] = total_outliers

        # 7. Reset index
        df = df.reset_index(drop=True)

        # Stats after cleaning
        for col in numeric_columns:
            if col in df.columns and len(df) > 0:
                report['column_stats_after'][col] = {
                    'count': int(df[col].notna().sum()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()) if len(df) > 1 else 0.0,
                }

        report['final_rows'] = len(df)
        report['removed_rows'] = report['original_rows'] - len(df)
        report['success'] = len(df) > 0
        self.report = report

        logger.info(
            f"Cleaning complete: {report['original_rows']} -> {report['final_rows']} rows "
            f"({report['removed_rows']} removed)"
        )
        return df, report

    def validate_required_columns(self, df: pd.DataFrame) -> Tuple[bool, Set[str]]:
        required = {'W', 'N', 'F', 'T', 'H', 'D', 'Formation', 'Wear_Type'}
        existing = set(df.columns.str.strip())
        missing = required - existing
        if missing:
            logger.warning(f"Missing columns: {missing}")
        else:
            logger.info("All required columns found")
        return len(missing) == 0, missing

    def generate_cleaning_report(self, report: Dict = None) -> str:
        if report is None:
            report = self.report

        lines = []
        lines.append("=" * 55)
        lines.append("         DATA CLEANING REPORT")
        lines.append("=" * 55)
        lines.append("")

        lines.append("[ SUMMARY ]")
        lines.append("-" * 40)
        lines.append(f"  Original rows      : {report.get('original_rows', 'N/A')}")
        lines.append(f"  Final rows         : {report.get('final_rows', 'N/A')}")
        lines.append(f"  Total removed      : {report.get('removed_rows', 'N/A')}")
        lines.append(f"    - Empty rows     : {report.get('empty_rows_removed', 0)}")
        lines.append(f"    - Duplicates     : {report.get('duplicates_removed', 0)}")
        lines.append(f"    - Outliers       : {report.get('outliers_removed', 0)}")
        lines.append(f"    - NaN rows       : {report.get('na_rows_removed', 0)}")
        lines.append(f"  Columns cleaned    : {', '.join(report.get('columns_cleaned', []))}")
        lines.append("")

        lines.append("[ ISSUES FOUND & FIXED ]")
        lines.append("-" * 40)
        issues = report.get('issues_found', [])
        if issues:
            for i, issue in enumerate(issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("  No issues found - data is clean!")
        lines.append("")

        stats_before = report.get('column_stats_before', {})
        if stats_before:
            lines.append("[ BEFORE CLEANING ]")
            lines.append("-" * 40)
            lines.append(f"  {'Col':<6} {'Count':>7} {'Miss':>6} {'Min':>10} {'Max':>10} {'Mean':>10}")
            for col, s in stats_before.items():
                mn = f"{s['min']:.2f}" if s.get('min') is not None else "N/A"
                mx = f"{s['max']:.2f}" if s.get('max') is not None else "N/A"
                me = f"{s['mean']:.2f}" if s.get('mean') is not None else "N/A"
                lines.append(
                    f"  {col:<6} {s.get('count', 0):>7} {s.get('missing', 0):>6} "
                    f"{mn:>10} {mx:>10} {me:>10}"
                )
            lines.append("")

        stats_after = report.get('column_stats_after', {})
        if stats_after:
            lines.append("[ AFTER CLEANING ]")
            lines.append("-" * 40)
            lines.append(f"  {'Col':<6} {'Count':>7} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
            for col, s in stats_after.items():
                lines.append(
                    f"  {col:<6} {s.get('count', 0):>7} "
                    f"{s.get('min', 0):>10.2f} {s.get('max', 0):>10.2f} "
                    f"{s.get('mean', 0):>10.2f} {s.get('std', 0):>10.2f}"
                )
            lines.append("")

        status = "SUCCESS" if report.get('success') else "FAILED"
        lines.append(f"[ STATUS: {status} ]")
        lines.append("=" * 55)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Drilling State & Model (unchanged math)
# ═══════════════════════════════════════════════════════════

@dataclass
class DrillingState:
    current_depth: float = 0.0
    cumulative_footage: float = 0.0
    cumulative_time: float = 0.0
    current_D: float = 1.0
    current_formation: str = "medium"
    current_W: float = 40000.0
    current_N: float = 120.0
    Af: float = 0.0
    Cf: float = 0.0
    last_F: float = -1.0
    last_T: float = -1.0
    history: List[Dict] = field(default_factory=list)

    def reset(self):
        self.current_depth = self.cumulative_footage = self.cumulative_time = 0.0
        self.current_D = 1.0
        self.Af = self.Cf = 0.0
        self.last_F = -1.0
        self.last_T = -1.0
        self.history = []
        logger.debug("DrillingState reset")


@dataclass
class MonteCarloResults:
    dfdT_all: np.ndarray = None
    dfdD_all: np.ndarray = None
    wob_valid: np.ndarray = None
    rpm_valid: np.ndarray = None
    time_all: np.ndarray = None
    footage_all: np.ndarray = None
    D_all: np.ndarray = None
    H_all: np.ndarray = None
    Af_all: np.ndarray = None
    Cf_all: np.ndarray = None
    W_star_all: np.ndarray = None
    M_star_all: np.ndarray = None
    M_all: np.ndarray = None
    R_all: np.ndarray = None
    a_all: np.ndarray = None
    a_power_p_all: np.ndarray = None
    U_all: np.ndarray = None
    Z_all: np.ndarray = None
    k_all: np.ndarray = None
    r_all: np.ndarray = None
    p_all: np.ndarray = None
    formation_all: np.ndarray = None
    wear_type_all: np.ndarray = None
    row_index_all: np.ndarray = None
    bootstrap_index_all: np.ndarray = None
    n_simulations: int = 0
    simulation_mode: str = "manual"
    W_range: tuple = None
    N_range: tuple = None


class GalleWoodsDynamicModel:
    FORMATIONS = {"soft": (0.95, 0.70), "medium": (1.00, 0.60), "hard": (1.05, 0.50)}
    WEAR_EXP = {"flat": 1.0, "sey": 0.5, "button_bits": 0.0, "custom": 0.75}

    def __init__(self):
        self.state = DrillingState()
        self.H = ModelConstants.DEFAULT_BIT_DIAMETER
        self.wear_type = "sey"
        self.wear_p = 0.5
        self.is_calibrated = False
        logger.info("GalleWoodsDynamicModel initialized")

    def _get_table_values(self, D, p):
        idx = max(1, min(int(math.floor(D)), 8))
        key = f"{idx}/8"
        U = TABLE_1_6[key]["U"]
        z_mapping = {1.0: "Z_1.0", 0.75: "Z_0.75", 0.5: "Z_0.5", 0.0: "Z_0"}
        Z = TABLE_1_6[key][z_mapping.get(p, "Z_0")]
        return U, Z

    def _calc_wear_factor_a(self, D):
        d = D / 8
        return ModelConstants.A_QUAD_COEF * d ** 2 + ModelConstants.A_LINEAR_COEF * d + ModelConstants.A_CONSTANT

    def _calc_W_star(self, W):
        if self.H <= 0:
            self.H = ModelConstants.DEFAULT_BIT_DIAMETER
        return max(ModelConstants.W_STAR_FACTOR * W / self.H, 0.001)

    def _calc_R(self, N):
        return N + ModelConstants.R_CUBIC_COEF * N ** 3

    def _calc_M_star(self, W_s):
        return max((ModelConstants.M_STAR_INTERCEPT - ModelConstants.M_STAR_SLOPE * math.log10(
            max(W_s, 0.001))) / ModelConstants.M_STAR_SLOPE, 0.001)

    def _calc_M_star_from_W(self, W):
        return max((ModelConstants.M_STAR_INTERCEPT - ModelConstants.M_STAR_SLOPE * math.log10(
            max(W, 0.001))) / ModelConstants.M_STAR_SLOPE, 0.001)

    def calculate_Af(self, T, N, W, D):
        if T <= 0:
            return 0.001
        U, Z = self._get_table_values(D, self.wear_p)
        W_s, R, M_s = self._calc_W_star(W), self._calc_R(N), self._calc_M_star(self._calc_W_star(W))
        return (T * R) / (M_s * U) if M_s * U > 0 else 0.001

    def calculate_Cf(self, F, T, W, N, D, formation, Af):
        if Af <= 0 or F <= 0:
            return 0
        k, r = self.FORMATIONS.get(formation.lower(), (1.0, 0.6))
        U, Z = self._get_table_values(D, self.wear_p)
        W_s, R, M_s = self._calc_W_star(W), self._calc_R(N), self._calc_M_star(self._calc_W_star(W))
        denom = M_s * (W_s ** k) * (N ** r) * Z * Af
        return (F * R) / denom if denom > 0 else 0

    def calculate_dfdT(self, W, N, Cf, D, formation):
        if Cf <= 0:
            return 0
        k, r = self.FORMATIONS.get(formation.lower(), (1.0, 0.6))
        a = self._calc_wear_factor_a(D)
        denom = a ** self.wear_p
        return max(0, (Cf * (W ** k) * (N ** r)) / denom) if denom > 0 else 0

    def calculate_dfdD(self, W, N, Cf, Af, D, formation):
        k, r = self.FORMATIONS.get(formation.lower(), (1.0, 0.6))
        R = self._calc_R(N)
        M = self._calc_M_star_from_W(W) * ModelConstants.M_STAR_SLOPE
        a = self._calc_wear_factor_a(D)
        return ((Cf * (W ** k) * (N ** r) * Af * M) / R) * (a ** (1 - self.wear_p)) if R > 0 else 0

    def calculate_all(self, W, N, F, T, H, D, formation, wear_type):
        self.H, self.wear_type = H, wear_type.lower()
        self.wear_p = self.WEAR_EXP.get(self.wear_type, 0.5)
        if not self.is_calibrated or F != self.state.last_F or T != self.state.last_T:
            self.state.Af = self.calculate_Af(T, N, W, D)
            self.state.Cf = self.calculate_Cf(F, T, W, N, D, formation.lower(), self.state.Af)
            self.state.last_F = F
            self.state.last_T = T
            self.is_calibrated = True
            logger.debug(f"Calibrated: Af={self.state.Af:.6f}, Cf={self.state.Cf:.6f}")

        dfdT = self.calculate_dfdT(W, N, self.state.Cf, D, formation.lower())
        dfdD = self.calculate_dfdD(W, N, self.state.Cf, self.state.Af, D, formation.lower())

        self.state.current_D, self.state.current_formation = D, formation.lower()
        self.state.current_W, self.state.current_N = W, N
        self.state.cumulative_footage, self.state.cumulative_time = F, T

        W_s, R, M_s = self._calc_W_star(W), self._calc_R(N), self._calc_M_star(self._calc_W_star(W))
        a = self._calc_wear_factor_a(D)
        U, Z = self._get_table_values(D, self.wear_p)
        k, r_exp = self.FORMATIONS.get(formation.lower(), (1.0, 0.6))

        logger.debug(f"Calculated: dF/dT={dfdT:.4f}, dF/dD={dfdD:.6f}, W={W}, N={N}")

        return {
            "Af": self.state.Af, "Cf": self.state.Cf,
            "dfdT": dfdT, "dfdD": dfdD,
            "W": W, "W_star": W_s, "M_star": M_s, "M": M_s * ModelConstants.M_STAR_SLOPE,
            "R": R, "a": a, "a_power_p": a ** self.wear_p, "U": U, "Z": Z,
            "k": k, "r": r_exp, "D": D, "p": self.wear_p
        }

    def reset(self):
        self.state.reset()
        self.is_calibrated = False
        logger.info("Model reset")


# ═══════════════════════════════════════════════════════════
# Monte Carlo Simulator (unchanged math)
# ═══════════════════════════════════════════════════════════

class MonteCarloSimulator:
    def __init__(self, model):
        self.model = model
        self.results = MonteCarloResults()
        logger.info("MonteCarloSimulator initialized")

    def run_simulation(self, params, n_simulations=1000):
        logger.info(f"Starting manual simulation: {n_simulations} iterations")

        W_min = params.get('W_min', 25000)
        W_max = params.get('W_max', 55000)
        N_min = params.get('N_min', 80)
        N_max = params.get('N_max', 160)
        F_fixed = params.get('F_fixed', 350)
        T_fixed = params.get('T_fixed', 10)
        D_fixed = params.get('D_fixed', 4)
        H_fixed = params.get('H', 12.25)
        formation = params.get('formation', 'medium')
        wear_type = params.get('wear_type', 'sey')

        logger.info(f"Params: W=[{W_min}-{W_max}], N=[{N_min}-{N_max}], F={F_fixed}, T={T_fixed}")

        wob_all = np.zeros(n_simulations)
        rpm_all = np.zeros(n_simulations)
        dfdT_all = np.zeros(n_simulations)
        dfdD_all = np.zeros(n_simulations)
        footage_all = np.zeros(n_simulations)
        time_all = np.zeros(n_simulations)
        D_all = np.zeros(n_simulations)
        H_all = np.zeros(n_simulations)
        Af_all = np.zeros(n_simulations)
        Cf_all = np.zeros(n_simulations)
        W_star_all = np.zeros(n_simulations)
        M_star_all = np.zeros(n_simulations)
        M_all = np.zeros(n_simulations)
        R_all = np.zeros(n_simulations)
        a_all = np.zeros(n_simulations)
        a_power_p_all = np.zeros(n_simulations)
        U_all = np.zeros(n_simulations)
        Z_all = np.zeros(n_simulations)
        k_all = np.zeros(n_simulations)
        r_all = np.zeros(n_simulations)
        p_all = np.zeros(n_simulations)
        formation_all = []
        wear_type_all = []
        error_count = 0

        for sim in range(n_simulations):
            W = np.random.uniform(W_min, W_max)
            N = np.random.uniform(N_min, N_max)
            try:
                res = self.model.calculate_all(W, N, F_fixed, T_fixed, H_fixed, D_fixed, formation, wear_type)
                dfdT_all[sim] = res['dfdT']
                dfdD_all[sim] = res['dfdD']
                Af_all[sim] = res['Af']
                Cf_all[sim] = res['Cf']
                W_star_all[sim] = res['W_star']
                M_star_all[sim] = res['M_star']
                M_all[sim] = res['M']
                R_all[sim] = res['R']
                a_all[sim] = res['a']
                a_power_p_all[sim] = res['a_power_p']
                U_all[sim] = res['U']
                Z_all[sim] = res['Z']
                k_all[sim] = res['k']
                r_all[sim] = res['r']
                p_all[sim] = res['p']
            except Exception as e:
                dfdT_all[sim] = 0
                dfdD_all[sim] = 0
                error_count += 1
                logger.warning(f"Sim #{sim}: error - {e}")

            wob_all[sim] = W
            rpm_all[sim] = N
            footage_all[sim] = F_fixed
            time_all[sim] = T_fixed
            D_all[sim] = D_fixed
            H_all[sim] = H_fixed
            formation_all.append(formation)
            wear_type_all.append(wear_type)

        valid = (dfdT_all > 0) & np.isfinite(dfdT_all) & np.isfinite(dfdD_all)
        valid_count = valid.sum()
        logger.info(f"Simulation done: {valid_count}/{n_simulations} valid, {error_count} errors")

        self.results = MonteCarloResults()
        self.results.wob_valid = wob_all[valid]
        self.results.rpm_valid = rpm_all[valid]
        self.results.time_all = time_all[valid]
        self.results.footage_all = footage_all[valid]
        self.results.D_all = D_all[valid]
        self.results.H_all = H_all[valid]
        self.results.formation_all = np.array(formation_all)[valid]
        self.results.wear_type_all = np.array(wear_type_all)[valid]
        self.results.dfdT_all = dfdT_all[valid]
        self.results.dfdD_all = dfdD_all[valid]
        self.results.Af_all = Af_all[valid]
        self.results.Cf_all = Cf_all[valid]
        self.results.W_star_all = W_star_all[valid]
        self.results.M_star_all = M_star_all[valid]
        self.results.M_all = M_all[valid]
        self.results.R_all = R_all[valid]
        self.results.a_all = a_all[valid]
        self.results.a_power_p_all = a_power_p_all[valid]
        self.results.U_all = U_all[valid]
        self.results.Z_all = Z_all[valid]
        self.results.k_all = k_all[valid]
        self.results.r_all = r_all[valid]
        self.results.p_all = p_all[valid]
        self.results.n_simulations = n_simulations
        self.results.simulation_mode = "manual"
        self.results.W_range = (W_min, W_max)
        self.results.N_range = (N_min, N_max)
        return self.results

    def run_simulation_from_data(self, df, n_bootstrap=1):
        logger.info(f"Starting file simulation: {len(df)} rows x {n_bootstrap} bootstrap")

        W_min_file, W_max_file = df['W'].min(), df['W'].max()
        N_min_file, N_max_file = df['N'].min(), df['N'].max()
        n_rows = len(df)
        total_sims = n_rows * n_bootstrap

        logger.info(f"File ranges: W=[{W_min_file:.0f}-{W_max_file:.0f}], N=[{N_min_file:.0f}-{N_max_file:.0f}]")

        wob_all = np.zeros(total_sims)
        rpm_all = np.zeros(total_sims)
        time_all = np.zeros(total_sims)
        footage_all = np.zeros(total_sims)
        D_all = np.zeros(total_sims)
        H_all = np.zeros(total_sims)
        formation_all = []
        wear_type_all = []
        dfdT_all = np.zeros(total_sims)
        dfdD_all = np.zeros(total_sims)
        Af_all = np.zeros(total_sims)
        Cf_all = np.zeros(total_sims)
        W_star_all = np.zeros(total_sims)
        M_star_all = np.zeros(total_sims)
        M_all = np.zeros(total_sims)
        R_all = np.zeros(total_sims)
        a_all = np.zeros(total_sims)
        a_power_p_all = np.zeros(total_sims)
        U_all = np.zeros(total_sims)
        Z_all = np.zeros(total_sims)
        k_all = np.zeros(total_sims)
        r_all = np.zeros(total_sims)
        p_all = np.zeros(total_sims)
        row_index_all = np.zeros(total_sims, dtype=int)
        bootstrap_index_all = np.zeros(total_sims, dtype=int)

        idx = 0
        error_count = 0

        for row_idx, (_, row) in enumerate(df.iterrows()):
            F, T, D, H = row['F'], row['T'], row['D'], row['H']
            formation = str(row['Formation']).lower()
            wear_type = str(row['Wear_Type']).lower()

            for boot in range(n_bootstrap):
                W = np.random.uniform(W_min_file, W_max_file)
                N = np.random.uniform(N_min_file, N_max_file)
                try:
                    res = self.model.calculate_all(W, N, F, T, H, D, formation, wear_type)
                    dfdT_all[idx] = res['dfdT']
                    dfdD_all[idx] = res['dfdD']
                    Af_all[idx] = res['Af']
                    Cf_all[idx] = res['Cf']
                    W_star_all[idx] = res['W_star']
                    M_star_all[idx] = res['M_star']
                    M_all[idx] = res['M']
                    R_all[idx] = res['R']
                    a_all[idx] = res['a']
                    a_power_p_all[idx] = res['a_power_p']
                    U_all[idx] = res['U']
                    Z_all[idx] = res['Z']
                    k_all[idx] = res['k']
                    r_all[idx] = res['r']
                    p_all[idx] = res['p']
                except Exception as e:
                    dfdT_all[idx] = 0
                    dfdD_all[idx] = 0
                    error_count += 1
                    logger.warning(f"Row {row_idx}, boot {boot}: {e}")

                wob_all[idx] = W
                rpm_all[idx] = N
                time_all[idx] = T
                footage_all[idx] = F
                D_all[idx] = D
                H_all[idx] = H
                formation_all.append(formation)
                wear_type_all.append(wear_type)
                row_index_all[idx] = row_idx + 1
                bootstrap_index_all[idx] = boot + 1
                idx += 1

        valid = (dfdT_all > 0) & np.isfinite(dfdT_all) & np.isfinite(dfdD_all)
        valid_count = valid.sum()
        logger.info(f"File simulation done: {valid_count}/{total_sims} valid, {error_count} errors")

        self.results = MonteCarloResults()
        self.results.wob_valid = wob_all[valid]
        self.results.rpm_valid = rpm_all[valid]
        self.results.time_all = time_all[valid]
        self.results.footage_all = footage_all[valid]
        self.results.D_all = D_all[valid]
        self.results.H_all = H_all[valid]
        self.results.formation_all = np.array(formation_all)[valid]
        self.results.wear_type_all = np.array(wear_type_all)[valid]
        self.results.row_index_all = row_index_all[valid]
        self.results.bootstrap_index_all = bootstrap_index_all[valid]
        self.results.dfdT_all = dfdT_all[valid]
        self.results.dfdD_all = dfdD_all[valid]
        self.results.Af_all = Af_all[valid]
        self.results.Cf_all = Cf_all[valid]
        self.results.W_star_all = W_star_all[valid]
        self.results.M_star_all = M_star_all[valid]
        self.results.M_all = M_all[valid]
        self.results.R_all = R_all[valid]
        self.results.a_all = a_all[valid]
        self.results.a_power_p_all = a_power_p_all[valid]
        self.results.U_all = U_all[valid]
        self.results.Z_all = Z_all[valid]
        self.results.k_all = k_all[valid]
        self.results.r_all = r_all[valid]
        self.results.p_all = p_all[valid]
        self.results.n_simulations = total_sims
        self.results.simulation_mode = "file"
        self.results.W_range = (W_min_file, W_max_file)
        self.results.N_range = (N_min_file, N_max_file)
        return self.results


# ═══════════════════════════════════════════════════════════
# UI Components
# ═══════════════════════════════════════════════════════════

class MetricCard(ctk.CTkFrame):
    def __init__(self, parent, title, value="--", unit="", color=COLORS["primary"], icon="📊"):
        super().__init__(parent, fg_color=COLORS["card"], corner_radius=12, border_width=1,
                         border_color=COLORS["border"])
        self.configure(height=100)
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(12, 5))
        ctk.CTkLabel(header, text=f"{icon} {title}", font=ctk.CTkFont(size=11),
                     text_color=COLORS["text_dim"]).pack(anchor="w")
        vf = ctk.CTkFrame(self, fg_color="transparent")
        vf.pack(fill="x", padx=15, pady=(0, 12))
        self.value_label = ctk.CTkLabel(vf, text=str(value), font=ctk.CTkFont(size=24, weight="bold"),
                                         text_color=color)
        self.value_label.pack(side="left")
        if unit:
            ctk.CTkLabel(vf, text=f" {unit}", font=ctk.CTkFont(size=12),
                         text_color=COLORS["text_dim"]).pack(side="left", pady=(8, 0))

    def update_value(self, value, color=None):
        self.value_label.configure(text=str(value))
        if color:
            self.value_label.configure(text_color=color)


# ═══════════════════════════════════════════════════════════
# Home Page
# ═══════════════════════════════════════════════════════════

class HomePage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=COLORS["light"])
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._create_info_section()
        logger.info("HomePage created")

    def _create_info_section(self):
        main = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=20, border_width=2,
                            border_color=COLORS["border"])
        main.grid(row=0, column=0, padx=40, pady=30, sticky="nsew")

        header = ctk.CTkFrame(main, fg_color=COLORS["primary"], corner_radius=12, height=60)
        header.pack(fill="x", padx=25, pady=(25, 20))
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="📋 Project Information", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="white").pack(expand=True)

        content = ctk.CTkScrollableFrame(main, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=25, pady=(0, 25))
        content.grid_columnconfigure((0, 1), weight=1)

        self.info_entries = {}
        fields_left = [("project_name", "🏷️ Project Name", ""), ("well_name", "🛢️ Well Name", ""),
                       ("field_name", "🏭 Field Name", ""), ("operator", "👷 Operator", "")]
        fields_right = [("location", "📍 Location", ""), ("rig_name", "🏗️ Rig Name", ""),
                        ("spud_date", "📅 Spud Date", datetime.now().strftime("%Y-%m-%d")),
                        ("target_depth", "🎯 Target Depth (ft)", "")]

        for i, (k, l, p) in enumerate(fields_left):
            self._create_field(content, k, l, p, i, 0)
        for i, (k, l, p) in enumerate(fields_right):
            self._create_field(content, k, l, p, i, 1)

        nf = ctk.CTkFrame(content, fg_color=COLORS["light"], corner_radius=12)
        nf.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")
        ctk.CTkLabel(nf, text="📝 Notes", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=15, pady=(12, 5))
        self.notes_text = ctk.CTkTextbox(nf, height=100, font=ctk.CTkFont(size=13), fg_color=COLORS["card"],
                                          border_color=COLORS["border"], border_width=1, corner_radius=10)
        self.notes_text.pack(fill="x", padx=15, pady=(0, 12))

        ic = ctk.CTkFrame(content, fg_color=COLORS["accent"], corner_radius=12)
        ic.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        ctk.CTkLabel(ic, text="📊 Monte Carlo - 9 Charts + AI Analysis",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="white").pack(anchor="w", padx=20,
                                                                                         pady=(15, 8))
        ctk.CTkLabel(ic,
                     text="• dF/dT vs W, N, T\n• dF/dD vs W, N, T\n• dF/dT & dF/dD Distributions\n• dF/dT vs dF/dD\n• 🤖 AI Safety Analysis with Gemini",
                     font=ctk.CTkFont(size=12), text_color="white", justify="left").pack(anchor="w", padx=20,
                                                                                          pady=(0, 15))

    def _create_field(self, parent, key, label, placeholder, row, col):
        f = ctk.CTkFrame(parent, fg_color=COLORS["light"], corner_radius=12)
        f.grid(row=row, column=col, padx=8, pady=8, sticky="ew")
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=15, pady=(12, 5))
        e = ctk.CTkEntry(f, height=45, font=ctk.CTkFont(size=13), placeholder_text=placeholder,
                         fg_color=COLORS["card"], border_color=COLORS["border"], border_width=1, corner_radius=10)
        e.pack(fill="x", padx=15, pady=(0, 12))
        self.info_entries[key] = e

    def get_project_info(self):
        info = {k: e.get() for k, e in self.info_entries.items()}
        info["notes"] = self.notes_text.get("1.0", "end-1c")
        return info


# ═══════════════════════════════════════════════════════════
# Config Frame - Enhanced with cleaning report
# ═══════════════════════════════════════════════════════════

class ConfigFrame(ctk.CTkFrame):
    def __init__(self, parent, model, app=None):
        super().__init__(parent, fg_color=COLORS["light"])
        self.model = model
        self.app = app
        self._update_pending = False
        self.processed_data = None
        self.loaded_df = None
        self.data_cleaner = DataCleaner()
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        self._create_params()
        self._create_results()
        logger.info("ConfigFrame created")

    def _create_params(self):
        left = ctk.CTkFrame(self, fg_color=COLORS["light"], corner_radius=20)
        left.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")

        header = ctk.CTkFrame(left, fg_color=COLORS["card"], corner_radius=15, height=70, border_width=1,
                              border_color=COLORS["border"])
        header.pack(fill="x", padx=15, pady=15)
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="⚙️ Drilling Parameters", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["primary"]).pack(side="left", padx=15)
        self.btn_export_excel = ctk.CTkButton(header, text="💾 Export", command=self.export_excel_data, width=90,
                                               height=35, state="disabled", fg_color=COLORS["accent"],
                                               hover_color="#0284c7")
        self.btn_export_excel.pack(side="right", padx=5)
        self.btn_load_excel = ctk.CTkButton(header, text="📂 Load", command=self.load_excel_data, width=90, height=35,
                                             fg_color=COLORS["success"], hover_color="#16a34a")
        self.btn_load_excel.pack(side="right", padx=5)

        ps = ctk.CTkScrollableFrame(left, fg_color="transparent")
        ps.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.entries = {}

        ds = ctk.CTkFrame(ps, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        ds.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(ds, text="🔧 Operational", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["success"]).pack(anchor="w", padx=15, pady=(10, 5))
        dg = ctk.CTkFrame(ds, fg_color="transparent")
        dg.pack(fill="x", padx=15, pady=10)
        dg.grid_columnconfigure((0, 1), weight=1)
        for i, (k, d, l) in enumerate([("W", "40000", "⚖️ WOB (lbs)"), ("N", "120", "🔄 RPM"),
                                        ("H", "12.25", "📏 Bit Dia (in)")]):
            self._entry(dg, k, d, l, i // 2, i % 2)

        pfs = ctk.CTkFrame(ps, fg_color=COLORS["card"], corner_radius=12, border_width=1,
                           border_color=COLORS["border"])
        pfs.pack(fill="x", padx=5, pady=10)
        ctk.CTkLabel(pfs, text="📊 Drilling Data", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["warning"]).pack(anchor="w", padx=15, pady=(10, 5))
        pg = ctk.CTkFrame(pfs, fg_color="transparent")
        pg.pack(fill="x", padx=15, pady=10)
        pg.grid_columnconfigure((0, 1), weight=1)
        for i, (k, d, l) in enumerate([("F", "368", "📐 Footage (ft)"), ("T", "10", "⏱️ Time (Hr)")]):
            self._entry(pg, k, d, l, 0, i)

        fs = ctk.CTkFrame(ps, fg_color=COLORS["card"], corner_radius=12, border_width=1,
                          border_color=COLORS["border"])
        fs.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(fs, text="🪨 Formation & Bit", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=15, pady=(10, 5))
        fg = ctk.CTkFrame(fs, fg_color="transparent")
        fg.pack(fill="x", padx=15, pady=10)
        fg.grid_columnconfigure((0, 1, 2), weight=1)
        self._combo(fg, "Wear_Fraction", "🔩 Wear Grade", list(TABLE_1_6.keys()), "4/8", 0, 0)
        self._combo(fg, "Formation Type", "🪨 Formation", ["Soft", "Medium", "Hard"], "Medium", 0, 1)
        self._combo(fg, "Wear Type", "⚡ Wear Type", ["Flat", "Sey", "Button_bits", "Custom"], "Sey", 0, 2)

        bf = ctk.CTkFrame(left, fg_color="transparent")
        bf.pack(fill="x", padx=15, pady=15)
        self.calc_btn = ctk.CTkButton(bf, text="🔬 CALCULATE", command=self.calculate, height=55,
                                       font=ctk.CTkFont(size=16, weight="bold"), fg_color=COLORS["primary"],
                                       hover_color=COLORS["secondary"], corner_radius=12)
        self.calc_btn.pack(fill="x")
        self.auto_calc_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(bf, text="🔄 Auto-Calculate", variable=self.auto_calc_var, font=ctk.CTkFont(size=12),
                        text_color=COLORS["text_dim"], fg_color=COLORS["primary"]).pack(pady=(10, 0))
        self.status_label = ctk.CTkLabel(bf, text="⏳ Ready", font=ctk.CTkFont(size=12, weight="bold"),
                                          text_color=COLORS["text_dim"])
        self.status_label.pack(pady=(10, 0))
        self.file_status = ctk.CTkLabel(bf, text="", font=ctk.CTkFont(size=11), text_color=COLORS["accent"])
        self.file_status.pack(pady=(5, 0))

    def _entry(self, p, k, d, l, r, c):
        f = ctk.CTkFrame(p, fg_color="transparent")
        f.grid(row=r, column=c, padx=8, pady=8, sticky="ew")
        ctk.CTkLabel(f, text=l, font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=COLORS["text_dim"]).pack(anchor="w")
        e = ctk.CTkEntry(f, height=45, font=ctk.CTkFont(size=14), fg_color=COLORS["lighter"],
                         border_color=COLORS["border"], border_width=2, corner_radius=10, justify="center")
        if d:
            e.insert(0, d)
        e.pack(fill="x", pady=(5, 0))
        e.bind("<Return>", lambda ev: self._sched() if self.auto_calc_var.get() else None)
        e.bind("<FocusOut>", lambda ev: self._sched() if self.auto_calc_var.get() else None)
        self.entries[k] = e

    def _combo(self, p, k, l, vals, d, r, c):
        f = ctk.CTkFrame(p, fg_color="transparent")
        f.grid(row=r, column=c, padx=8, pady=8, sticky="ew")
        ctk.CTkLabel(f, text=l, font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=COLORS["text_dim"]).pack(anchor="w")
        cb = ctk.CTkComboBox(f, values=vals, height=45, font=ctk.CTkFont(size=13), fg_color=COLORS["lighter"],
                             border_color=COLORS["border"], button_color=COLORS["primary"], corner_radius=10,
                             command=lambda _: self._sched() if self.auto_calc_var.get() else None)
        cb.set(d)
        cb.pack(fill="x", pady=(5, 0))
        self.entries[k] = cb

    def _sched(self):
        if not self._update_pending:
            self._update_pending = True
            self.after(100, self._do_update)

    def _do_update(self):
        self._update_pending = False
        self.calculate()

    def load_excel_data(self):
        logger.info("User initiated file load")
        fp = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if not fp:
            logger.info("File load cancelled")
            return

        try:
            logger.info(f"Loading file: {fp}")
            df = pd.read_excel(fp)
            logger.info(f"File loaded: {len(df)} rows, columns={list(df.columns)}")

            is_valid, missing_cols = self.data_cleaner.validate_required_columns(df)
            if not is_valid:
                logger.error(f"Missing columns: {missing_cols}")
                messagebox.showerror("❌ Missing Columns", f"Missing: {missing_cols}")
                return

            cleaned_df, report = self.data_cleaner.clean_dataframe(df)
            cleaning_report_text = self.data_cleaner.generate_cleaning_report(report)
            logger.info("Cleaning report generated:\n" + cleaning_report_text)

            if not report['success']:
                logger.error("Cleaning failed - no valid data")
                messagebox.showerror("❌ Cleaning Failed", "No valid data!\n\n" + cleaning_report_text)
                return

            res = []
            for idx, row in cleaned_df.iterrows():
                try:
                    result = self.model.calculate_all(
                        float(row['W']), float(row['N']), float(row['F']), float(row['T']),
                        float(row['H']), float(row['D']), str(row['Formation']), str(row['Wear_Type'])
                    )
                    res.append(result)
                    logger.debug(f"Row {idx}: dF/dT={result['dfdT']:.4f}, dF/dD={result['dfdD']:.6f}")
                except Exception as e:
                    res.append({"Error": str(e)})
                    logger.error(f"Row {idx} error: {e}")

            self.processed_data = pd.concat([cleaned_df, pd.DataFrame(res)], axis=1)
            self.processed_data = self.processed_data.loc[:, ~self.processed_data.columns.duplicated()]
            self.loaded_df = cleaned_df

            self.btn_export_excel.configure(state="normal")
            self.status_label.configure(text=f"✅ {len(cleaned_df)} rows loaded", text_color=COLORS["success"])
            self.file_status.configure(text="📁 Ready for Monte Carlo", text_color=COLORS["accent"])

            if self.app and hasattr(self.app, 'frames') and 'monte_carlo' in self.app.frames:
                self.app.frames['monte_carlo'].set_loaded_data(cleaned_df)

            logger.info(f"File processing complete: {len(cleaned_df)} valid rows")

            # Show cleaning report window
            self._show_cleaning_report_window(cleaning_report_text, report)

        except Exception as e:
            logger.error(f"File load error: {e}", exc_info=True)
            messagebox.showerror("❌ Error", str(e))

    def _show_cleaning_report_window(self, report_text: str, report: Dict):
        win = ctk.CTkToplevel(self)
        win.title("📊 Data Cleaning Report")
        win.geometry("650x550")
        win.transient(self)
        win.grab_set()

        header = ctk.CTkFrame(win, fg_color=COLORS["primary"], corner_radius=0, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="📊 Data Cleaning Report",
                     font=ctk.CTkFont(size=16, weight="bold"), text_color="white").pack(expand=True)

        summary_frame = ctk.CTkFrame(win, fg_color=COLORS["light"], corner_radius=10)
        summary_frame.pack(fill="x", padx=15, pady=(10, 5))
        summary_grid = ctk.CTkFrame(summary_frame, fg_color="transparent")
        summary_grid.pack(fill="x", padx=10, pady=8)
        summary_grid.grid_columnconfigure((0, 1, 2, 3), weight=1)

        for i, (label, value, color) in enumerate([
            ("📥 Original", str(report.get('original_rows', 0)), COLORS["primary"]),
            ("📤 Final", str(report.get('final_rows', 0)), COLORS["success"]),
            ("🗑️ Removed", str(report.get('removed_rows', 0)), COLORS["danger"]),
            ("🔧 Filled", str(report.get('missing_filled', 0)), COLORS["warning"]),
        ]):
            sf = ctk.CTkFrame(summary_grid, fg_color=COLORS["card"], corner_radius=8)
            sf.grid(row=0, column=i, padx=4, pady=4, sticky="ew")
            ctk.CTkLabel(sf, text=label, font=ctk.CTkFont(size=10),
                         text_color=COLORS["text_dim"]).pack(pady=(6, 2))
            ctk.CTkLabel(sf, text=value, font=ctk.CTkFont(size=18, weight="bold"),
                         text_color=color).pack(pady=(0, 6))

        text_frame = ctk.CTkFrame(win, fg_color=COLORS["card"], corner_radius=10)
        text_frame.pack(fill="both", expand=True, padx=15, pady=5)
        report_textbox = ctk.CTkTextbox(text_frame, font=ctk.CTkFont(family="Courier", size=11),
                                         fg_color=COLORS["lighter"], corner_radius=8)
        report_textbox.pack(fill="both", expand=True, padx=8, pady=8)
        report_textbox.insert("1.0", report_text)
        report_textbox.configure(state="disabled")

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkButton(btn_frame, text="✅ OK", command=win.destroy, width=120, height=40,
                      fg_color=COLORS["primary"], font=ctk.CTkFont(size=13, weight="bold")).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="📋 Copy Report",
                      command=lambda: self._copy_report_clipboard(report_text, win),
                      width=120, height=40, fg_color=COLORS["accent"],
                      font=ctk.CTkFont(size=13, weight="bold")).pack(side="right", padx=5)

    def _copy_report_clipboard(self, text, window):
        window.clipboard_clear()
        window.clipboard_append(text)
        logger.info("Cleaning report copied to clipboard")
        messagebox.showinfo("✅ Copied", "Report copied to clipboard!")

    def export_excel_data(self):
        if self.processed_data is None:
            logger.warning("Export attempted with no data")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if fp:
            try:
                export_df = self.processed_data.copy()
                if 'dfdT' in export_df.columns:
                    export_df = export_df.rename(columns={'dfdT': 'dF/dT'})
                if 'dfdD' in export_df.columns:
                    export_df = export_df.rename(columns={'dfdD': 'dF/dD'})
                export_df.to_excel(fp, index=False)
                logger.info(f"Data exported to: {fp}")
                messagebox.showinfo("Success", f"Exported to:\n{fp}")
            except Exception as e:
                logger.error(f"Export error: {e}")
                messagebox.showerror("Error", str(e))

    def _create_results(self):
        right = ctk.CTkFrame(self, fg_color=COLORS["light"], corner_radius=20)
        right.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")

        header = ctk.CTkFrame(right, fg_color=COLORS["card"], corner_radius=15, height=70, border_width=1,
                              border_color=COLORS["border"])
        header.pack(fill="x", padx=15, pady=15)
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="📊 Results", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLORS["primary"]).pack(side="left", padx=15)

        mr = ctk.CTkFrame(right, fg_color="transparent")
        mr.pack(fill="x", padx=15, pady=10)
        mr.grid_columnconfigure((0, 1), weight=1)
        self.metric_cards = {}
        for i, (k, t, u, c, ico) in enumerate([
            ("dfdT", "dF/dT", "ft/hr", COLORS["success"], "🚀"),
            ("dfdD", "dF/dD", "", COLORS["accent"], "📈"),
            ("Cf", "Cf", "", COLORS["warning"], "🪨"),
            ("Af", "Af", "", COLORS["primary"], "⚙️"),
            ("M", "M", "", COLORS["danger"], "📐"),
            ("M_s", "M*", "", COLORS["secondary"], "📊")
        ]):
            self.metric_cards[k] = MetricCard(mr, t, "--", u, c, ico)
            self.metric_cards[k].grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="ew")

        df = ctk.CTkFrame(right, fg_color=COLORS["card"], corner_radius=12, border_width=1,
                          border_color=COLORS["border"])
        df.pack(fill="both", expand=True, padx=15, pady=10)
        ctk.CTkLabel(df, text="📋 Details", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=15, pady=(10, 5))
        self.result_labels = {}
        rs = ctk.CTkScrollableFrame(df, fg_color="transparent")
        rs.pack(fill="both", expand=True, padx=10, pady=10)
        for k, d in [("W", "W (lbs)"), ("W_star", "W*"), ("a", "a"), ("a_power_p", "a^p"),
                     ("R", "R"), ("k", "k"), ("r", "r"), ("p", "p"), ("U", "U"), ("Z", "Z")]:
            row = ctk.CTkFrame(rs, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=d, font=ctk.CTkFont(size=11), text_color=COLORS["text_dim"]).pack(side="left")
            lbl = ctk.CTkLabel(row, text="--", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text"])
            lbl.pack(side="right")
            self.result_labels[k] = lbl

    def get_values(self):
        try:
            v = {}
            for k, w in self.entries.items():
                if isinstance(w, ctk.CTkEntry):
                    val = w.get().strip()
                    if not val:
                        return None
                    v[k] = float(val)
                else:
                    v[k] = w.get()
            return v
        except Exception as e:
            logger.error(f"Error reading values: {e}")
            return None

    def calculate(self):
        logger.info("Manual calculation triggered")
        vals = self.get_values()
        if not vals:
            logger.warning("Calculation aborted: empty fields")
            messagebox.showerror("Error", "Fill all fields!")
            return None
        try:
            D = int(vals["Wear_Fraction"].split('/')[0])
            res = self.model.calculate_all(vals["W"], vals["N"], vals["F"], vals["T"], vals["H"], D,
                                            vals["Formation Type"], vals["Wear Type"])
            logger.info(f"Result: dF/dT={res['dfdT']:.4f}, dF/dD={res['dfdD']:.6f}, Cf={res['Cf']:.8f}")

            self.status_label.configure(text="✅ Done", text_color=COLORS["success"])
            self.metric_cards["dfdT"].update_value(f"{res['dfdT']:.2f}")
            self.metric_cards["dfdD"].update_value(f"{res['dfdD']:.4f}")
            self.metric_cards["Cf"].update_value(f"{res['Cf']:.6f}")
            self.metric_cards["Af"].update_value(f"{res['Af']:.6f}")
            self.metric_cards["M"].update_value(f"{res['M']:.4f}")
            self.metric_cards["M_s"].update_value(f"{res['M_star']:.6f}")
            for k, lbl in self.result_labels.items():
                if k in res:
                    lbl.configure(text=f"{res[k]:.4f}" if isinstance(res[k], float) else str(res[k]))
            return res
        except Exception as e:
            logger.error(f"Calculation error: {e}", exc_info=True)
            messagebox.showerror("Error", str(e))
            return None


# ═══════════════════════════════════════════════════════════
# Monte Carlo Page (unchanged chart/AI logic)
# ═══════════════════════════════════════════════════════════

class MonteCarloPage(ctk.CTkFrame):
    def __init__(self, parent, model, config_frame=None):
        super().__init__(parent, fg_color=COLORS["light"])
        self.model = model
        self.simulator = MonteCarloSimulator(model)
        self.ai_analyzer = GeminiDrillingAnalyzer()
        self.results = None
        self.ai_analysis = None
        self.loaded_data = None
        self.config_frame = config_frame

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)

        self._create_controls()
        self._create_charts()
        logger.info("MonteCarloPage created")

    def set_config_frame(self, config_frame):
        self.config_frame = config_frame

    def set_loaded_data(self, df):
        self.loaded_data = df
        W_min, W_max = df['W'].min(), df['W'].max()
        N_min, N_max = df['N'].min(), df['N'].max()
        for entry, val in [(self.W_min_entry, W_min), (self.W_max_entry, W_max),
                           (self.N_min_entry, N_min), (self.N_max_entry, N_max)]:
            entry.delete(0, 'end')
            entry.insert(0, f"{val:.0f}")
        self.data_status_label.configure(
            text=f"✅ {len(df)} rows | W:[{W_min:.0f}-{W_max:.0f}] N:[{N_min:.0f}-{N_max:.0f}]",
            text_color=COLORS["success"])
        self.run_from_file_btn.configure(state="normal")
        logger.info(f"Loaded data set: {len(df)} rows")

    def _get_config_values(self):
        if self.config_frame is None:
            return None
        try:
            vals = self.config_frame.get_values()
            if vals is None:
                return None
            D = int(vals["Wear_Fraction"].split('/')[0])
            return {'F': vals.get('F', 350), 'T': vals.get('T', 10), 'D': D,
                    'H': vals.get('H', 12.25),
                    'formation': vals.get('Formation Type', 'Medium').lower(),
                    'wear_type': vals.get('Wear Type', 'Sey').lower()}
        except Exception as e:
            logger.error(f"Error getting config values: {e}")
            return None

    def _create_controls(self):
        left = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=15,
                            border_width=1, border_color=COLORS["border"])
        left.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="nsew")

        header = ctk.CTkFrame(left, fg_color=COLORS["monte_carlo_ci"], corner_radius=12, height=50)
        header.pack(fill="x", padx=15, pady=(15, 10))
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="🎲 Monte Carlo Simulation",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color="white").pack(expand=True)

        info_frame = ctk.CTkFrame(left, fg_color=COLORS["light"], corner_radius=10)
        info_frame.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkLabel(info_frame, text="ℹ️ Fixed values from Config page",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["primary"]).pack(pady=(8, 2))
        self.config_status_label = ctk.CTkLabel(info_frame, text="⚙️ F, T, D, H, Formation, Wear Type",
                                                 font=ctk.CTkFont(size=9), text_color=COLORS["text_dim"])
        self.config_status_label.pack(pady=(0, 4))
        self.data_status_label = ctk.CTkLabel(info_frame, text="📁 No file loaded",
                                               font=ctk.CTkFont(size=9), text_color=COLORS["warning"])
        self.data_status_label.pack(pady=(0, 8))

        ps = ctk.CTkScrollableFrame(left, fg_color="transparent", height=250)
        ps.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        random_section = ctk.CTkFrame(ps, fg_color=COLORS["primary"], corner_radius=12)
        random_section.pack(fill="x", padx=5, pady=(5, 8))
        ctk.CTkLabel(random_section, text="🎲 Random Variables (W, N)",
                     font=ctk.CTkFont(size=11, weight="bold"), text_color="white").pack(anchor="w", padx=12,
                                                                                         pady=(10, 8))

        iter_frame = ctk.CTkFrame(random_section, fg_color="transparent")
        iter_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(iter_frame, text="🔢 Number of Simulations:",
                     font=ctk.CTkFont(size=10), text_color="white").pack(anchor="w")
        self.n_sim_entry = ctk.CTkEntry(iter_frame, height=32, font=ctk.CTkFont(size=12),
                                         justify="center", fg_color=COLORS["lighter"],
                                         border_color=COLORS["border"], corner_radius=8)
        self.n_sim_entry.insert(0, "1000")
        self.n_sim_entry.pack(fill="x", pady=(4, 0))

        self.W_min_entry, self.W_max_entry = self._range_inputs(random_section, "⚖️ WOB Range (lbs)", "25000", "55000")
        self.N_min_entry, self.N_max_entry = self._range_inputs(random_section, "🔄 RPM Range", "80", "160")

        bootstrap_section = ctk.CTkFrame(ps, fg_color=COLORS["accent"], corner_radius=12)
        bootstrap_section.pack(fill="x", padx=5, pady=(0, 8))
        boot_inner = ctk.CTkFrame(bootstrap_section, fg_color="transparent")
        boot_inner.pack(fill="x", padx=12, pady=10)
        ctk.CTkLabel(boot_inner, text="🔁 Simulations per row:",
                     font=ctk.CTkFont(size=10), text_color="white").pack(anchor="w")
        self.bootstrap_entry = ctk.CTkEntry(boot_inner, height=32, font=ctk.CTkFont(size=12),
                                             justify="center", fg_color=COLORS["lighter"], corner_radius=6)
        self.bootstrap_entry.insert(0, "1")
        self.bootstrap_entry.pack(fill="x", pady=(5, 0))

        buttons_frame = ctk.CTkFrame(left, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=15, pady=(5, 8))

        self.run_btn = ctk.CTkButton(buttons_frame, text="🚀 RUN MANUAL", command=self.run_simulation,
                                      height=42, font=ctk.CTkFont(size=12, weight="bold"),
                                      fg_color=COLORS["primary"], hover_color=COLORS["secondary"], corner_radius=10)
        self.run_btn.pack(fill="x", pady=(0, 6))

        self.run_from_file_btn = ctk.CTkButton(buttons_frame, text="📁 RUN FROM FILE",
                                                command=self.run_simulation_from_file, height=42,
                                                font=ctk.CTkFont(size=12, weight="bold"),
                                                fg_color=COLORS["success"], hover_color="#16a34a",
                                                corner_radius=10, state="disabled")
        self.run_from_file_btn.pack(fill="x", pady=(0, 6))

        self.ai_analyze_btn = ctk.CTkButton(buttons_frame, text="🤖 AI SAFETY ANALYSIS",
                                             command=self.run_ai_analysis, height=42,
                                             font=ctk.CTkFont(size=12, weight="bold"),
                                             fg_color="#8b5cf6", hover_color="#7c3aed",
                                             corner_radius=10, state="disabled")
        self.ai_analyze_btn.pack(fill="x", pady=(0, 6))

        self.export_btn = ctk.CTkButton(buttons_frame, text="💾 EXPORT RESULTS", command=self.export_results,
                                         height=38, font=ctk.CTkFont(size=11, weight="bold"),
                                         fg_color=COLORS["warning"], hover_color="#d97706",
                                         state="disabled", corner_radius=10)
        self.export_btn.pack(fill="x")

        progress_frame = ctk.CTkFrame(left, fg_color=COLORS["light"], corner_radius=10)
        progress_frame.pack(fill="x", padx=15, pady=(5, 15))
        self.progress_label = ctk.CTkLabel(progress_frame, text="⏳ Ready to simulate",
                                            font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["text_dim"])
        self.progress_label.pack(pady=(8, 4))
        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=6, fg_color=COLORS["border"],
                                                progress_color=COLORS["success"], corner_radius=3)
        self.progress_bar.pack(fill="x", padx=12, pady=(0, 8))
        self.progress_bar.set(0)

        self.ai_info_frame = ctk.CTkFrame(left, fg_color=COLORS["light"], corner_radius=10)
        self.ai_info_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.ai_status_label = ctk.CTkLabel(
            self.ai_info_frame,
            text="🤖 AI: Ready" if self.ai_analyzer.is_configured else "⚠️ AI: API Key needed",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=COLORS["success"] if self.ai_analyzer.is_configured else COLORS["warning"])
        self.ai_status_label.pack(pady=8)

    def _range_inputs(self, parent, label, default_min, default_max):
        frm = ctk.CTkFrame(parent, fg_color="transparent")
        frm.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(frm, text=label, font=ctk.CTkFont(size=10), text_color="white").pack(anchor="w")
        row = ctk.CTkFrame(frm, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        row.grid_columnconfigure((0, 1, 2), weight=1)

        mf = ctk.CTkFrame(row, fg_color="transparent")
        mf.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkLabel(mf, text="Min", font=ctk.CTkFont(size=8), text_color="#d0d0d0").pack(anchor="w")
        e_min = ctk.CTkEntry(mf, height=30, font=ctk.CTkFont(size=11), justify="center",
                             fg_color=COLORS["lighter"], corner_radius=6)
        e_min.insert(0, default_min)
        e_min.pack(fill="x")

        ctk.CTkLabel(row, text="→", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="white").grid(row=0, column=1, pady=(12, 0))

        xf = ctk.CTkFrame(row, fg_color="transparent")
        xf.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        ctk.CTkLabel(xf, text="Max", font=ctk.CTkFont(size=8), text_color="#d0d0d0").pack(anchor="w")
        e_max = ctk.CTkEntry(xf, height=30, font=ctk.CTkFont(size=11), justify="center",
                             fg_color=COLORS["lighter"], corner_radius=6)
        e_max.insert(0, default_max)
        e_max.pack(fill="x")
        return e_min, e_max

    def _create_charts(self):
        right = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=15,
                             border_width=1, border_color=COLORS["border"])
        right.grid(row=0, column=1, padx=(10, 15), pady=15, sticky="nsew")

        header_frame = ctk.CTkFrame(right, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 5))
        ctk.CTkLabel(header_frame, text="📊 Monte Carlo Results - 9 Charts",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["primary"]).pack(side="left")
        self.source_indicator = ctk.CTkLabel(header_frame, text="",
                                              font=ctk.CTkFont(size=10), text_color=COLORS["text_dim"])
        self.source_indicator.pack(side="right", padx=10)

        self.charts_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.charts_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.fig = Figure(figsize=(16, 10), dpi=100, facecolor=COLORS["chart_bg"])
        self.fig.subplots_adjust(hspace=0.35, wspace=0.28, left=0.06, right=0.98, top=0.94, bottom=0.06)
        self.axes = [self.fig.add_subplot(3, 3, i + 1) for i in range(9)]
        for ax in self.axes:
            ax.set_facecolor(COLORS["chart_bg"])
            ax.grid(True, alpha=0.25, color=COLORS["grid_color"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._setup_plots()

    def _setup_plots(self):
        titles = ["1. dF/dT vs W", "2. dF/dT vs N", "3. dF/dT vs T",
                  "4. dF/dD vs W", "5. dF/dD vs N", "6. dF/dD vs T",
                  "7. dF/dT Distribution", "8. dF/dD Distribution", "9. dF/dT vs dF/dD"]
        for ax, t in zip(self.axes, titles):
            ax.clear()
            ax.set_title(t, fontsize=9, fontweight="bold", color=COLORS["text"])
            ax.set_facecolor(COLORS["chart_bg"])
            ax.grid(True, alpha=0.25, color=COLORS["grid_color"])
            ax.text(0.5, 0.5, "Run simulation to see results", ha="center", va="center",
                    fontsize=9, color=COLORS["text_dim"], transform=ax.transAxes)
        self.canvas.draw()

    def get_simulation_params(self):
        try:
            params = {
                'n_simulations': float(self.n_sim_entry.get()),
                'W_min': float(self.W_min_entry.get()),
                'W_max': float(self.W_max_entry.get()),
                'N_min': float(self.N_min_entry.get()),
                'N_max': float(self.N_max_entry.get()),
            }
            config_vals = self._get_config_values()
            if config_vals:
                params.update({'F_fixed': config_vals['F'], 'T_fixed': config_vals['T'],
                              'D_fixed': config_vals['D'], 'H': config_vals['H'],
                              'formation': config_vals['formation'], 'wear_type': config_vals['wear_type']})
                self.config_status_label.configure(
                    text=f"✅ F={config_vals['F']}, T={config_vals['T']}, D={config_vals['D']}, H={config_vals['H']}",
                    text_color=COLORS["success"])
            else:
                params.update({'F_fixed': 350, 'T_fixed': 10, 'D_fixed': 4,
                              'H': 12.25, 'formation': 'medium', 'wear_type': 'sey'})
                self.config_status_label.configure(text="⚠️ Using defaults", text_color=COLORS["warning"])
            logger.info(f"Simulation params: {params}")
            return params
        except ValueError as e:
            logger.error(f"Invalid params: {e}")
            messagebox.showerror("❌ Input Error", f"Please enter valid values!\n{e}")
            return None

    def run_simulation(self):
        logger.info("Manual simulation started")
        params = self.get_simulation_params()
        if params is None:
            return
        try:
            self.progress_label.configure(text="🔄 Running...", text_color=COLORS["warning"])
            self.progress_bar.set(0.1)
            self.update()
            n_sims = int(params.get('n_simulations', 1000))
            self.progress_bar.set(0.3)
            self.update()
            self.results = self.simulator.run_simulation(params, n_sims)
            self.ai_analysis = None
            self.progress_bar.set(0.7)
            self.update()
            self._update_plots()
            self.progress_bar.set(1.0)
            valid_count = len(self.results.dfdT_all) if self.results else 0
            self.progress_label.configure(text=f"✅ {valid_count:,} valid results", text_color=COLORS["success"])
            self.source_indicator.configure(text=f"📊 Manual | F={params.get('F_fixed')}, T={params.get('T_fixed')}")
            self.export_btn.configure(state="normal")
            self.ai_analyze_btn.configure(state="normal")
            logger.info(f"Manual simulation complete: {valid_count} valid results")
        except Exception as e:
            self.progress_label.configure(text="❌ Error", text_color=COLORS["danger"])
            self.progress_bar.set(0)
            logger.error(f"Simulation error: {e}", exc_info=True)
            messagebox.showerror("Error", str(e))

    def run_simulation_from_file(self):
        logger.info("File simulation started")
        if self.loaded_data is None:
            messagebox.showwarning("⚠️ No Data", "Load data from Config first!")
            return
        try:
            self.progress_label.configure(text="🔄 Running from File...", text_color=COLORS["warning"])
            self.progress_bar.set(0.1)
            self.update()
            try:
                n_bootstrap = int(self.bootstrap_entry.get())
            except:
                n_bootstrap = 1
            self.progress_bar.set(0.3)
            self.update()
            self.results = self.simulator.run_simulation_from_data(self.loaded_data, n_bootstrap)
            self.ai_analysis = None
            self.progress_bar.set(0.7)
            self.update()
            self._update_plots()
            self.progress_bar.set(1.0)
            valid_count = len(self.results.dfdT_all) if self.results else 0
            self.progress_label.configure(text=f"✅ {valid_count:,} valid results", text_color=COLORS["success"])
            self.source_indicator.configure(text=f"📁 File Mode ({len(self.loaded_data)} rows)")
            self.export_btn.configure(state="normal")
            self.ai_analyze_btn.configure(state="normal")
            logger.info(f"File simulation complete: {valid_count} valid results")
        except Exception as e:
            self.progress_label.configure(text="❌ Error", text_color=COLORS["danger"])
            self.progress_bar.set(0)
            logger.error(f"File simulation error: {e}", exc_info=True)
            messagebox.showerror("Error", str(e))

    def run_ai_analysis(self):
        logger.info("AI analysis started")
        if self.results is None or len(self.results.dfdT_all) == 0:
            messagebox.showwarning("⚠️ No Data", "Run simulation first!")
            return
        if not self.ai_analyzer.is_configured:
            messagebox.showerror("❌ AI Not Configured", "Gemini API key is not configured!")
            return
        try:
            self.progress_label.configure(text="🤖 Running AI Analysis...", text_color="#8b5cf6")
            self.progress_bar.set(0.2)
            self.ai_status_label.configure(text="🔄 Analyzing...", text_color=COLORS["warning"])
            self.update()

            self.ai_analysis = self.ai_analyzer.analyze_drilling_data(self.results)
            self.progress_bar.set(0.6)
            self.update()

            self._update_plots_with_safety()
            self.progress_bar.set(1.0)

            best = self.ai_analysis.get('best_point', {})
            top_3 = self.ai_analysis.get('top_3_options', {})
            n_optimal = len(self.ai_analysis.get('optimal_indices', []))
            rejection = self.ai_analysis.get('rejection_analysis', {})

            self.progress_label.configure(text=f"🤖 AI Complete | Optimal: {n_optimal} points", text_color="#8b5cf6")
            self.ai_status_label.configure(
                text=f"✅ Best: W={best.get('wob', 0):.0f}, N={best.get('rpm', 0):.0f}",
                text_color=COLORS["success"])

            logger.info(f"AI analysis complete: {n_optimal} optimal points, best WOB={best.get('wob', 0):.0f}")

            report_lines = [
                "=" * 50, "🤖 AI SAFETY ANALYSIS REPORT", "=" * 50, "",
                "📌 SELECTION CRITERIA:",
                "   1st Priority: Highest dF/dT ",
                "   2nd Priority: Highest dF/dd ", "",
                "📊 CLASSIFICATION SUMMARY:",
                f"   🟣 Optimal Points: {rejection.get('optimal_count', n_optimal)}",
                f"   🟢 Safe Points: {rejection.get('safe_count', 0)}",
                f"   🟡 Caution Points: {rejection.get('caution_count', 0)}",
                f"   🔴 Unsafe Points: {rejection.get('unsafe_count', 0)}", "",
                "=" * 50, "🎯 TOP 3 OPTIMAL OPTIONS:", "=" * 50,
            ]

            if top_3:
                for name, label in [('option_1', '🥇 OPTION 1 (BEST)'), ('option_2', '🥈 OPTION 2'),
                                    ('option_3', '🥉 OPTION 3')]:
                    opt = top_3.get(name)
                    if opt:
                        report_lines.extend(["", f"{label}:",
                                             f"   • WOB: {opt['wob']:.0f} lbs",
                                             f"   • RPM: {opt['rpm']:.0f}",
                                             f"   • dF/dD: {opt['dfdD']:.6f}",
                                             f"   • dF/dT: {opt['dfdT']:.2f} ft/hr"])

            report_lines.extend(["", "=" * 50,
                                 "❌ REJECTION ANALYSIS:", "=" * 50,
                                 f"   🔴 Unsafe ({rejection.get('unsafe_count', 0)}):",
                                 f"      {rejection.get('reason_unsafe', 'N/A')}", "",
                                 f"   🟡 Caution ({rejection.get('caution_count', 0)}):",
                                 f"      {rejection.get('reason_caution', 'N/A')}", "",
                                 "=" * 50, "💡 AI RECOMMENDATION:", "=" * 50,
                                 f"   {self.ai_analysis.get('ai_recommendation', 'N/A')}", "",
                                 "🎨 LEGEND:",
                                 "   🟢 Green = Safe | 🟡 Yellow = Caution",
                                 "   🔴 Red = Unsafe | 🟣 Purple = Optimal",
                                 "   ⭐ Gold Star = Best point", "=" * 50])

            self._show_report_window("\n".join(report_lines))
        except Exception as e:
            self.progress_label.configure(text="❌ AI Error", text_color=COLORS["danger"])
            self.ai_status_label.configure(text="❌ Error", text_color=COLORS["danger"])
            self.progress_bar.set(0)
            logger.error(f"AI Analysis error: {e}", exc_info=True)
            messagebox.showerror("AI Error", str(e))

    def _show_report_window(self, report_text):
        report_window = ctk.CTkToplevel(self)
        report_window.title("🤖 AI Analysis Report")
        report_window.geometry("700x800")
        report_window.grab_set()

        header = ctk.CTkFrame(report_window, fg_color="#8b5cf6", corner_radius=0, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="🤖 AI Drilling Safety Analysis Report",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="white").pack(expand=True)

        text_frame = ctk.CTkFrame(report_window, fg_color=COLORS["card"])
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)
        report_textbox = ctk.CTkTextbox(text_frame, font=ctk.CTkFont(family="Consolas", size=11),
                                         fg_color=COLORS["lighter"], text_color=COLORS["text"])
        report_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        report_textbox.insert("1.0", report_text)
        report_textbox.configure(state="disabled")

        btn_frame = ctk.CTkFrame(report_window, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkButton(btn_frame, text="📋 Copy", command=lambda: self._copy_to_clipboard(report_text),
                      fg_color=COLORS["primary"]).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="💾 Save", command=lambda: self._save_report(report_text),
                      fg_color=COLORS["success"]).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="✖️ Close", command=report_window.destroy,
                      fg_color=COLORS["danger"]).pack(side="right", padx=5)

    def _copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        logger.info("AI report copied to clipboard")
        messagebox.showinfo("✅ Copied", "Report copied to clipboard!")

    def _save_report(self, text):
        fp = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if fp:
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"AI report saved to: {fp}")
            messagebox.showinfo("✅ Saved", f"Report saved to:\n{fp}")

    def _get_color_array(self, safety_colors):
        color_map = {'safe': COLORS["safe"], 'caution': COLORS["caution"],
                     'unsafe': COLORS["unsafe"], 'optimal': COLORS["optimal"]}
        return [color_map.get(c, COLORS["text_dim"]) for c in safety_colors]

    def _update_plots_with_safety(self):
        if self.results is None or self.ai_analysis is None:
            return
        logger.info("Updating charts with AI safety colors")

        r = self.results
        analysis = self.ai_analysis
        colors = self._get_color_array(analysis['safety_colors'])
        best_idx = analysis.get('best_point_index', 0)

        for ax in self.axes:
            ax.clear()

        # Charts 1-6: Scatter with safety colors
        chart_configs = [
            (0, r.wob_valid, r.dfdT_all, "W (lbs)", "dF/dT (ft/hr)", "1. dF/dT vs W"),
            (1, r.rpm_valid, r.dfdT_all, "N (RPM)", "dF/dT (ft/hr)", "2. dF/dT vs N"),
            (2, r.time_all, r.dfdT_all, "T (hr)", "dF/dT (ft/hr)", "3. dF/dT vs T"),
            (3, r.wob_valid, r.dfdD_all, "W (lbs)", "dF/dD", "4. dF/dD vs W"),
            (4, r.rpm_valid, r.dfdD_all, "N (RPM)", "dF/dD", "5. dF/dD vs N"),
            (5, r.time_all, r.dfdD_all, "T (hr)", "dF/dD", "6. dF/dD vs T"),
        ]
        for idx, x, y, xl, yl, title in chart_configs:
            self.axes[idx].scatter(x, y, c=colors, alpha=0.6, s=12)
            self.axes[idx].scatter([x[best_idx]], [y[best_idx]], c='gold', s=200, marker='*',
                                   edgecolors='black', linewidths=1, zorder=5)
            self.axes[idx].set_xlabel(xl, fontsize=8)
            self.axes[idx].set_ylabel(yl, fontsize=8)
            self.axes[idx].set_title(title, fontsize=9, fontweight="bold")

        # Charts 7-8: Horizontal histograms with safety
        for chart_idx, data_arr, ylabel, title in [
            (6, r.dfdT_all, "dF/dT (ft/hr)", "7. dF/dT Distribution"),
            (7, r.dfdD_all, "dF/dD", "8. dF/dD Distribution")
        ]:
            for mask_name, color, label in [
                ('safe', COLORS["safe"], 'Safe'), ('caution', COLORS["caution"], 'Caution'),
                ('unsafe', COLORS["unsafe"], 'Unsafe'), ('optimal', COLORS["optimal"], 'Optimal')
            ]:
                mask = analysis['safety_colors'] == mask_name
                if np.any(mask):
                    self.axes[chart_idx].hist(data_arr[mask], bins=20, color=color, alpha=0.7,
                                              label=label, orientation='horizontal')
            self.axes[chart_idx].legend(fontsize=6, loc='upper right')
            self.axes[chart_idx].set_xlabel("Frequency", fontsize=8)
            self.axes[chart_idx].set_ylabel(ylabel, fontsize=8)
            self.axes[chart_idx].set_title(title, fontsize=9, fontweight="bold")

        # Chart 9: Cross-plot
        self.axes[8].scatter(r.dfdD_all, r.dfdT_all, c=colors, alpha=0.6, s=12)
        self.axes[8].scatter([r.dfdD_all[best_idx]], [r.dfdT_all[best_idx]], c='gold', s=200, marker='*',
                             edgecolors='black', linewidths=1, zorder=5, label='Best Point')
        self.axes[8].legend(fontsize=7, loc='upper right')
        self.axes[8].set_xlabel("dF/dD", fontsize=8)
        self.axes[8].set_ylabel("dF/dT (ft/hr)", fontsize=8)
        self.axes[8].set_title("9. dF/dT vs dF/dD (Safety)", fontsize=9, fontweight="bold")

        for ax in self.axes:
            ax.set_facecolor(COLORS["chart_bg"])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=7)

        self.fig.tight_layout()
        self.canvas.draw()
        logger.info("Charts updated with safety colors")

    def _update_plots(self):
        if self.results is None or len(self.results.dfdT_all) == 0:
            return
        logger.info("Updating 9 charts")

        r = self.results
        for ax in self.axes:
            ax.clear()

        self.axes[0].scatter(r.wob_valid, r.dfdT_all, alpha=0.4, s=8, c=COLORS["dfdT_color"])
        self.axes[0].set_xlabel("W (lbs)", fontsize=8)
        self.axes[0].set_ylabel("dF/dT (ft/hr)", fontsize=8)
        self.axes[0].set_title("1. dF/dT vs W", fontsize=9, fontweight="bold")

        self.axes[1].scatter(r.rpm_valid, r.dfdT_all, alpha=0.4, s=8, c=COLORS["rpm_color"])
        self.axes[1].set_xlabel("N (RPM)", fontsize=8)
        self.axes[1].set_ylabel("dF/dT (ft/hr)", fontsize=8)
        self.axes[1].set_title("2. dF/dT vs N", fontsize=9, fontweight="bold")

        self.axes[2].scatter(r.time_all, r.dfdT_all, alpha=0.4, s=8, c=COLORS["time_color"])
        self.axes[2].set_xlabel("T (hr)", fontsize=8)
        self.axes[2].set_ylabel("dF/dT (ft/hr)", fontsize=8)
        self.axes[2].set_title("3. dF/dT vs T", fontsize=9, fontweight="bold")

        self.axes[3].scatter(r.wob_valid, r.dfdD_all, alpha=0.4, s=8, c=COLORS["dfdD_color"])
        self.axes[3].set_xlabel("W (lbs)", fontsize=8)
        self.axes[3].set_ylabel("dF/dD", fontsize=8)
        self.axes[3].set_title("4. dF/dD vs W", fontsize=9, fontweight="bold")

        self.axes[4].scatter(r.rpm_valid, r.dfdD_all, alpha=0.4, s=8, c=COLORS["warning"])
        self.axes[4].set_xlabel("N (RPM)", fontsize=8)
        self.axes[4].set_ylabel("dF/dD", fontsize=8)
        self.axes[4].set_title("5. dF/dD vs N", fontsize=9, fontweight="bold")

        self.axes[5].scatter(r.time_all, r.dfdD_all, alpha=0.4, s=8, c=COLORS["accent"])
        self.axes[5].set_xlabel("T (hr)", fontsize=8)
        self.axes[5].set_ylabel("dF/dD", fontsize=8)
        self.axes[5].set_title("6. dF/dD vs T", fontsize=9, fontweight="bold")

        # Chart 7: Horizontal histogram
        self.axes[6].hist(r.dfdT_all, bins=40, color=COLORS["dfdT_color"], alpha=0.75,
                          edgecolor='white', linewidth=0.5, orientation='horizontal')
        self.axes[6].set_xlabel("Frequency", fontsize=8)
        self.axes[6].set_ylabel("dF/dT (ft/hr)", fontsize=8)
        self.axes[6].set_title("7. dF/dT Distribution", fontsize=9, fontweight="bold")

        # Chart 8: Horizontal histogram
        self.axes[7].hist(r.dfdD_all, bins=40, color=COLORS["dfdD_color"], alpha=0.75,
                          edgecolor='white', linewidth=0.5, orientation='horizontal')
        self.axes[7].set_xlabel("Frequency", fontsize=8)
        self.axes[7].set_ylabel("dF/dD", fontsize=8)
        self.axes[7].set_title("8. dF/dD Distribution", fontsize=9, fontweight="bold")

        self.axes[8].scatter(r.dfdD_all, r.dfdT_all, alpha=0.4, s=8, c=COLORS["monte_carlo_ci"])
        self.axes[8].set_xlabel("dF/dD", fontsize=8)
        self.axes[8].set_ylabel("dF/dT (ft/hr)", fontsize=8)
        self.axes[8].set_title("9. dF/dT vs dF/dD", fontsize=9, fontweight="bold")

        for ax in self.axes:
            ax.set_facecolor(COLORS["chart_bg"])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=7)

        self.fig.tight_layout()
        self.canvas.draw()
        logger.info("Charts updated successfully")

    def export_results(self):
        if self.results is None:
            messagebox.showwarning("⚠️", "Run simulation first!")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not fp:
            return
        try:
            r = self.results
            logger.info(f"Exporting results to: {fp}")

            summary_df = pd.DataFrame({
                'Parameter': ['Total Simulations', 'Valid Results', 'Mode', 'WOB Range', 'RPM Range',
                              'Export Date', 'AI Analysis'],
                'Value': [r.n_simulations, len(r.dfdT_all), r.simulation_mode,
                          f"{r.W_range[0]:.0f} - {r.W_range[1]:.0f}" if r.W_range else "N/A",
                          f"{r.N_range[0]:.0f} - {r.N_range[1]:.0f}" if r.N_range else "N/A",
                          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          'Yes' if self.ai_analysis else 'No']
            })

            simulation_df = pd.DataFrame({
                'No': range(1, len(r.dfdT_all) + 1),
                'WOB': r.wob_valid, 'RPM': r.rpm_valid,
                'dF/dT': r.dfdT_all, 'dF/dD': r.dfdD_all,
                'F': r.footage_all, 'T': r.time_all,
                'D': r.D_all, 'H': r.H_all,
                'Formation': r.formation_all, 'Wear_Type': r.wear_type_all,
                'Af': r.Af_all, 'Cf': r.Cf_all,
                'W_star': r.W_star_all, 'M_star': r.M_star_all,
                'M': r.M_all, 'R': r.R_all,
                'a': r.a_all, 'a^p': r.a_power_p_all,
                'U': r.U_all, 'Z': r.Z_all,
                'k': r.k_all, 'r': r.r_all, 'p': r.p_all
            })

            if self.ai_analysis:
                simulation_df['Safety_Class'] = self.ai_analysis['safety_colors']
            if r.row_index_all is not None and len(r.row_index_all) > 0:
                simulation_df['Source_Row'] = r.row_index_all
            if r.bootstrap_index_all is not None and len(r.bootstrap_index_all) > 0:
                simulation_df['Bootstrap'] = r.bootstrap_index_all

            with pd.ExcelWriter(fp, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                simulation_df.to_excel(writer, sheet_name='Simulation_Data', index=False)

                if self.ai_analysis:
                    best = self.ai_analysis.get('best_point', {})
                    top_3 = self.ai_analysis.get('top_3_options', {})
                    rejection = self.ai_analysis.get('rejection_analysis', {})
                    ai_data = [
                        ['BEST POINT', '', ''],
                        ['WOB', best.get('wob', 0), 'lbs'],
                        ['RPM', best.get('rpm', 0), ''],
                        ['dF/dD', best.get('dfdD', 0), ''],
                        ['dF/dT', best.get('dfdT', 0), 'ft/hr'],
                        ['', '', ''],
                        ['CLASSIFICATION', '', ''],
                        ['Optimal', rejection.get('optimal_count', 0), ''],
                        ['Safe', rejection.get('safe_count', 0), ''],
                        ['Caution', rejection.get('caution_count', 0), ''],
                        ['Unsafe', rejection.get('unsafe_count', 0), ''],
                    ]
                    if top_3:
                        ai_data.append(['', '', ''])
                        ai_data.append(['TOP 3 OPTIONS', '', ''])
                        for i, name in enumerate(['option_1', 'option_2', 'option_3']):
                            opt = top_3.get(name)
                            if opt:
                                ai_data.extend([
                                    [f'OPTION {i + 1}', '', ''],
                                    ['  WOB', opt['wob'], 'lbs'],
                                    ['  RPM', opt['rpm'], ''],
                                    ['  dF/dD', opt['dfdD'], ''],
                                    ['  dF/dT', opt['dfdT'], 'ft/hr'],
                                ])
                    pd.DataFrame(ai_data, columns=['Metric', 'Value', 'Unit']).to_excel(
                        writer, sheet_name='AI_Analysis', index=False)

            logger.info(f"Export successful: {len(r.dfdT_all)} records to {fp}")
            messagebox.showinfo("✅ Export Success", f"Exported:\n{fp}\n\n{len(r.dfdT_all):,} records saved")
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            messagebox.showerror("❌ Export Error", str(e))


# ═══════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════

class DrillingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        logger.info("Initializing DrillingApp")

        self.title("🛢️ Galle-Woods Drilling Model")
        self.geometry("1650x980")
        self.minsize(1400, 850)

        self.model = GalleWoodsDynamicModel()
        self.ai_analyzer = GeminiDrillingAnalyzer()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._create_sidebar()

        self.main_container = ctk.CTkFrame(self, fg_color=COLORS["light"], corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.frames = {}
        self._create_frames()
        self.show_frame("home")
        logger.info("DrillingApp initialization complete")

    def _create_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=COLORS["card"], width=200, corner_radius=0,
                               border_width=1, border_color=COLORS["border"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        logo_frame = ctk.CTkFrame(sidebar, fg_color=COLORS["primary"], corner_radius=12, height=80)
        logo_frame.pack(fill="x", padx=12, pady=(15, 20))
        logo_frame.pack_propagate(False)
        ctk.CTkLabel(logo_frame, text="🛢️ Drilling Simulator",
                     font=ctk.CTkFont(size=15, weight="bold"), text_color="white").pack(expand=True)

        self.nav_btns = {}
        for text, key in [("🏠 Home", "home"), ("⚙️ Configuration", "config"), ("🎲 Monte Carlo", "monte_carlo")]:
            btn = ctk.CTkButton(sidebar, text=text, command=lambda k=key: self.show_frame(k),
                                height=50, font=ctk.CTkFont(size=13, weight="bold"),
                                fg_color="transparent", hover_color=COLORS["light"],
                                text_color=COLORS["text"], anchor="w", corner_radius=10)
            btn.pack(fill="x", padx=12, pady=4)
            self.nav_btns[key] = btn

        sep = ctk.CTkFrame(sidebar, fg_color=COLORS["border"], height=2)
        sep.pack(fill="x", padx=20, pady=15)

        ai_status_frame = ctk.CTkFrame(sidebar, fg_color=COLORS["light"], corner_radius=10)
        ai_status_frame.pack(fill="x", padx=12, pady=5)
        ctk.CTkLabel(ai_status_frame, text="🤖 Gemini AI Status",
                     font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["text"]).pack(pady=(10, 5))
        status_text = "🟢 Connected" if self.ai_analyzer.is_configured else "🔴 Not Configured"
        status_color = COLORS["success"] if self.ai_analyzer.is_configured else COLORS["danger"]
        ctk.CTkLabel(ai_status_frame, text=status_text, font=ctk.CTkFont(size=10),
                     text_color=status_color).pack(pady=(0, 10))

        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=12, pady=15)
        ctk.CTkLabel(footer, text="v2.5 - AI Enhanced", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_dim"]).pack()
        ctk.CTkLabel(footer, text="© 2024", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_dim"]).pack()

    def _create_frames(self):
        self.frames["home"] = HomePage(self.main_container)
        self.frames["config"] = ConfigFrame(self.main_container, self.model, app=self)
        self.frames["monte_carlo"] = MonteCarloPage(self.main_container, self.model,
                                                     config_frame=self.frames["config"])
        self.frames["monte_carlo"].ai_analyzer = self.ai_analyzer
        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")
        logger.info("All frames created")

    def show_frame(self, name):
        logger.debug(f"Switching to frame: {name}")
        for key, btn in self.nav_btns.items():
            if key == name:
                btn.configure(fg_color=COLORS["primary"], text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text"])
        if name in self.frames:
            self.frames[name].tkraise()


if __name__ == "__main__":
    logger.info("Application starting...")
    app = DrillingApp()
    app.mainloop()
    logger.info("Application closed")