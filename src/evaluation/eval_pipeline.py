"""
RAGAS Evaluation Pipeline for RAG system quality assessment.
Evaluates faithfulness, answer relevance, context precision, and context recall.
"""
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import time
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from loguru import logger

from src.agents.rag_orchestrator import RAGOrchestrator
from src.config import EVALUATION_DIR


class RAGASEvaluationPipeline:
    """
    RAGAS-based evaluation pipeline for RAG system.

    Metrics:
    - Faithfulness: Answer is grounded in context (no hallucinations)
    - Answer Relevancy: Answer addresses the question
    - Context Recall: All relevant info is in retrieved context
    - Context Precision: Retrieved context is relevant
    """

    def __init__(
        self,
        rag_orchestrator: RAGOrchestrator,
        golden_dataset_path: Optional[Path] = None
    ):
        """
        Initialize evaluation pipeline.

        Args:
            rag_orchestrator: RAG orchestrator instance
            golden_dataset_path: Path to golden dataset JSON
        """
        self.rag_orchestrator = rag_orchestrator
        self.logger = logger.bind(module="eval_pipeline")

        if golden_dataset_path is None:
            golden_dataset_path = EVALUATION_DIR / "golden_dataset.json"

        self.golden_dataset_path = golden_dataset_path
        self.golden_dataset = self._load_golden_dataset()

        self.logger.info(f"Evaluation pipeline initialized with {len(self.golden_dataset)} test cases")

    def _load_golden_dataset(self) -> List[Dict[str, Any]]:
        """
        Load golden dataset from JSON.

        Returns:
            List of test cases
        """
        try:
            with open(self.golden_dataset_path, 'r') as f:
                dataset = json.load(f)

            self.logger.info(f"Loaded {len(dataset)} test cases from {self.golden_dataset_path}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading golden dataset: {str(e)}")
            return []

    def run_system_on_dataset(
        self,
        top_k: int = 5,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run the RAG system on all golden dataset questions.

        Args:
            top_k: Number of chunks to retrieve
            save_results: Whether to save results to file

        Returns:
            List of system outputs
        """
        self.logger.info(f"Running system on {len(self.golden_dataset)} questions...")

        results = []

        for i, test_case in enumerate(self.golden_dataset, 1):
            question = test_case["question"]
            question_id = test_case.get("question_id", f"Q{i}")

            self.logger.info(f"[{i}/{len(self.golden_dataset)}] Processing {question_id}")

            try:
                # Get system answer
                start_time = time.time()

                system_output = self.rag_orchestrator.answer_question(
                    question=question,
                    top_k_final=top_k
                )

                processing_time = time.time() - start_time

                # Extract data for evaluation
                result = {
                    "question_id": question_id,
                    "question": question,
                    "answer": system_output.get("answer", ""),
                    "contexts": self._extract_contexts(system_output),
                    "ground_truth": test_case.get("expected_answer", ""),
                    "processing_time_ms": int(processing_time * 1000),
                    "calculation_steps": system_output.get("calculation_steps"),
                    "citations": system_output.get("citations", []),
                    "confidence": system_output.get("confidence", 0.0),
                    "metadata": {
                        "page_reference": test_case.get("page_reference"),
                        "section": test_case.get("section"),
                        "requires_calculation": test_case.get("requires_calculation", False),
                        "difficulty": test_case.get("difficulty", "unknown"),
                        "tags": test_case.get("tags", [])
                    }
                }

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {question_id}: {str(e)}")
                results.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": f"ERROR: {str(e)}",
                    "contexts": [],
                    "ground_truth": test_case.get("expected_answer", ""),
                    "error": str(e)
                })

            # Clear calculation history between queries
            self.rag_orchestrator.clear_calculation_history()

        # Save results
        if save_results:
            self._save_results(results, "system_outputs")

        return results

    def _extract_contexts(self, system_output: Dict[str, Any]) -> List[str]:
        """
        Extract context strings from system output.

        Args:
            system_output: RAG system output

        Returns:
            List of context strings
        """
        contexts = []

        citations = system_output.get("citations", [])

        for citation in citations:
            if isinstance(citation, dict):
                context = citation.get("text", "")
            else:
                context = str(citation)

            if context:
                contexts.append(context)

        return contexts

    def evaluate_with_ragas(
        self,
        system_results: List[Dict[str, Any]],
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate system results using RAGAS metrics.

        Args:
            system_results: Results from run_system_on_dataset()
            metrics: List of RAGAS metrics to use

        Returns:
            Evaluation results
        """
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]

        self.logger.info(f"Evaluating {len(system_results)} results with RAGAS...")

        # Prepare data for RAGAS
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        for result in system_results:
            # Skip error results
            if "error" in result:
                continue

            ragas_data["question"].append(result["question"])
            ragas_data["answer"].append(result["answer"])
            ragas_data["contexts"].append(result["contexts"])
            ragas_data["ground_truth"].append(result["ground_truth"])

        # Create HuggingFace dataset
        dataset = Dataset.from_dict(ragas_data)

        self.logger.info(f"Dataset prepared: {len(dataset)} samples")

        # Run RAGAS evaluation
        try:
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=metrics
            )

            # Convert to dict
            results_dict = {
                "overall_scores": evaluation_result.to_pandas().mean().to_dict(),
                "per_question_scores": evaluation_result.to_pandas().to_dict('records'),
                "summary": {
                    "num_questions": len(dataset),
                    "timestamp": datetime.now().isoformat(),
                    "metrics_used": [m.name for m in metrics]
                }
            }

            self.logger.info("RAGAS evaluation completed")
            self.logger.info(f"Overall scores: {results_dict['overall_scores']}")

            return results_dict

        except Exception as e:
            self.logger.error(f"Error during RAGAS evaluation: {str(e)}")
            return {
                "error": str(e),
                "overall_scores": {},
                "per_question_scores": []
            }

    def run_full_evaluation(
        self,
        top_k: int = 5,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            top_k: Number of chunks to retrieve
            save_report: Whether to save evaluation report

        Returns:
            Complete evaluation report
        """
        self.logger.info("Starting full evaluation pipeline...")

        # Step 1: Run system on dataset
        system_results = self.run_system_on_dataset(top_k=top_k)

        # Step 2: Evaluate with RAGAS
        ragas_results = self.evaluate_with_ragas(system_results)

        # Step 3: Compute additional metrics
        additional_metrics = self._compute_additional_metrics(system_results)

        # Step 4: Create comprehensive report
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "system_configuration": {
                "top_k": top_k,
                "num_test_cases": len(self.golden_dataset),
                "num_completed": len([r for r in system_results if "error" not in r])
            },
            "ragas_metrics": ragas_results,
            "additional_metrics": additional_metrics,
            "system_results": system_results
        }

        # Save report
        if save_report:
            self._save_results(report, "evaluation_report")

        self.logger.info("Full evaluation completed")

        return report

    def _compute_additional_metrics(
        self,
        system_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute additional custom metrics.

        Args:
            system_results: System outputs

        Returns:
            Additional metrics
        """
        total = len(system_results)
        errors = len([r for r in system_results if "error" in r])
        successful = total - errors

        # Processing times
        times = [r.get("processing_time_ms", 0) for r in system_results if "error" not in r]
        avg_time = sum(times) / len(times) if times else 0

        # Calculation accuracy (for questions requiring calculations)
        calc_questions = [r for r in system_results if r.get("metadata", {}).get("requires_calculation")]
        calc_with_steps = len([r for r in calc_questions if r.get("calculation_steps")])

        # Confidence scores
        confidences = [r.get("confidence", 0) for r in system_results if "error" not in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Difficulty breakdown
        difficulty_breakdown = {}
        for result in system_results:
            difficulty = result.get("metadata", {}).get("difficulty", "unknown")
            if difficulty not in difficulty_breakdown:
                difficulty_breakdown[difficulty] = {"total": 0, "errors": 0}

            difficulty_breakdown[difficulty]["total"] += 1
            if "error" in result:
                difficulty_breakdown[difficulty]["errors"] += 1

        return {
            "success_rate": successful / total if total > 0 else 0,
            "error_rate": errors / total if total > 0 else 0,
            "avg_processing_time_ms": avg_time,
            "avg_confidence": avg_confidence,
            "calculation_tool_usage": {
                "questions_requiring_calc": len(calc_questions),
                "questions_with_calc_steps": calc_with_steps,
                "usage_rate": calc_with_steps / len(calc_questions) if calc_questions else 0
            },
            "difficulty_breakdown": difficulty_breakdown
        }

    def _save_results(self, results: Any, filename: str):
        """
        Save results to JSON file.

        Args:
            results: Results to save
            filename: Base filename (without extension)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = EVALUATION_DIR / f"{filename}_{timestamp}.json"

        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            self.logger.info(f"Results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def generate_report_summary(self, report: Dict[str, Any]) -> str:
        """
        Generate human-readable report summary.

        Args:
            report: Evaluation report

        Returns:
            Formatted summary string
        """
        ragas = report.get("ragas_metrics", {}).get("overall_scores", {})
        additional = report.get("additional_metrics", {})

        summary = f"""
========================================
RAG SYSTEM EVALUATION REPORT
========================================
Timestamp: {report.get('evaluation_timestamp', 'N/A')}

RAGAS METRICS:
--------------
Faithfulness:       {ragas.get('faithfulness', 0):.3f}
Answer Relevancy:   {ragas.get('answer_relevancy', 0):.3f}
Context Precision:  {ragas.get('context_precision', 0):.3f}
Context Recall:     {ragas.get('context_recall', 0):.3f}

ADDITIONAL METRICS:
-------------------
Success Rate:       {additional.get('success_rate', 0):.1%}
Avg Processing:     {additional.get('avg_processing_time_ms', 0):.0f}ms
Avg Confidence:     {additional.get('avg_confidence', 0):.3f}
Calc Tool Usage:    {additional.get('calculation_tool_usage', {}).get('usage_rate', 0):.1%}

TEST CASES:
-----------
Total:              {report.get('system_configuration', {}).get('num_test_cases', 0)}
Completed:          {report.get('system_configuration', {}).get('num_completed', 0)}

========================================
"""

        return summary


def create_evaluation_pipeline(
    rag_orchestrator: RAGOrchestrator,
    golden_dataset_path: Optional[Path] = None
) -> RAGASEvaluationPipeline:
    """
    Factory function to create evaluation pipeline.

    Args:
        rag_orchestrator: RAG orchestrator instance
        golden_dataset_path: Path to golden dataset

    Returns:
        RAGASEvaluationPipeline instance
    """
    return RAGASEvaluationPipeline(
        rag_orchestrator=rag_orchestrator,
        golden_dataset_path=golden_dataset_path
    )


if __name__ == "__main__":
    print("RAGAS Evaluation Pipeline module loaded successfully")
    print("Use create_evaluation_pipeline() to create an instance")
