#!/usr/bin/env python3
"""Run evaluation of anomaly detection system."""

import argparse
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_evaluation(args):
    """Run the evaluation."""
    logger.info("Starting AeroSentry evaluation")
    
    # Import components
    from shared.eval.synthetic_attacks import create_attack_dataset
    from shared.eval.metrics import evaluate_detector, evaluate_by_attack_type, EvaluationReport
    from edge.edge_inference.rules import RuleEngine
    from edge.edge_inference.anomaly_model import TrackAnomalyDetector, EnsembleAnomalyDetector
    from edge.features.track_features import TrackWindow
    
    # Create synthetic dataset
    logger.info(f"Generating synthetic dataset: {args.n_normal} normal, {args.n_attacks} attacks")
    
    dataset = create_attack_dataset(
        n_normal=args.n_normal,
        n_spoofed=args.n_attacks // 4,
        n_replay=args.n_attacks // 4,
        n_ghost=args.n_attacks // 4,
        n_saturation=args.n_attacks - 3 * (args.n_attacks // 4)
    )
    
    # Initialize detectors
    rule_engine = RuleEngine()
    ml_detector = TrackAnomalyDetector(contamination=0.1)
    ensemble = EnsembleAnomalyDetector()
    
    # Extract features from tracks
    logger.info("Extracting features from tracks...")
    
    features_list = []
    labels = []
    attack_types = []
    
    for track, label, attack_type in zip(
        dataset["tracks"],
        dataset["labels"],
        dataset.get("attack_types", ["unknown"] * len(dataset["labels"]))
    ):
        # Create track window
        window = TrackWindow(
            icao24=track[0].get("icao24", "test"),
            max_points=1000,
            window_seconds=3600
        )
        
        for point in track:
            window.add_point(point)
            
        if window.can_compute():
            features = window.compute_features()
            features_list.append(features)
            labels.append(label)
            attack_types.append(attack_type)
            
    logger.info(f"Extracted features from {len(features_list)} tracks")
    
    # Train ML detector on normal data
    logger.info("Training ML detector...")
    normal_features = [f for f, l in zip(features_list, labels) if l == 0]
    ml_detector.fit(normal_features)
    
    # Generate predictions
    logger.info("Generating predictions...")
    
    rule_scores = []
    ml_scores = []
    ensemble_scores = []
    
    for features, icao24 in zip(features_list, [f"test{i}" for i in range(len(features_list))]):
        # Rule-based
        triggers = rule_engine.evaluate(features, icao24)
        rule_score = max((t.severity.value / 4.0 for t in triggers), default=0)
        rule_scores.append(rule_score)
        
        # ML-based
        ml_score = ml_detector.predict(features)
        ml_scores.append(ml_score)
        
        # Ensemble
        combined = ensemble.combine_scores(rule_score, ml_score)
        ensemble_scores.append(combined)
        
    # Evaluate
    logger.info("Evaluating detectors...")
    
    print("\n" + "=" * 60)
    print("AEROSENTRY EVALUATION RESULTS")
    print("=" * 60)
    
    # Rule-based results
    print("\n--- Rule-Based Detector ---")
    rule_results = evaluate_detector(rule_scores, labels, threshold=0.25)
    print(f"AUC-ROC: {rule_results['auc_roc']:.3f}")
    print(f"Precision: {rule_results['precision']:.3f}")
    print(f"Recall: {rule_results['recall']:.3f}")
    print(f"F1-Score: {rule_results['f1_score']:.3f}")
    
    # ML results
    print("\n--- ML Detector (Isolation Forest) ---")
    ml_results = evaluate_detector(ml_scores, labels, threshold=0.5)
    print(f"AUC-ROC: {ml_results['auc_roc']:.3f}")
    print(f"Precision: {ml_results['precision']:.3f}")
    print(f"Recall: {ml_results['recall']:.3f}")
    print(f"F1-Score: {ml_results['f1_score']:.3f}")
    
    # Ensemble results
    print("\n--- Ensemble Detector ---")
    ensemble_results = evaluate_detector(ensemble_scores, labels, threshold=0.4)
    print(f"AUC-ROC: {ensemble_results['auc_roc']:.3f}")
    print(f"Precision: {ensemble_results['precision']:.3f}")
    print(f"Recall: {ensemble_results['recall']:.3f}")
    print(f"F1-Score: {ensemble_results['f1_score']:.3f}")
    
    # Per-attack-type results
    if len(set(attack_types)) > 1:
        print("\n--- Results by Attack Type ---")
        by_type = evaluate_by_attack_type(ensemble_scores, labels, attack_types)
        for attack_type, results in by_type.items():
            if attack_type != "normal":
                print(f"\n{attack_type}:")
                print(f"  Recall: {results['recall']:.3f}")
                print(f"  Precision: {results['precision']:.3f}")
                
    # Generate report
    report = EvaluationReport()
    report.add_detector("Rule-Based", rule_results)
    report.add_detector("Isolation Forest", ml_results)
    report.add_detector("Ensemble", ensemble_results)
    
    if args.output:
        report_path = Path(args.output)
        report_path.write_text(report.to_markdown())
        logger.info(f"Report saved to {report_path}")
        
    print("\n" + "=" * 60)
    print("Evaluation complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run AeroSentry evaluation")
    parser.add_argument(
        "--n-normal",
        type=int,
        default=100,
        help="Number of normal tracks"
    )
    parser.add_argument(
        "--n-attacks",
        type=int,
        default=40,
        help="Number of attack tracks"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for evaluation report"
    )
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
