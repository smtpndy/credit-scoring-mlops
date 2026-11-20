"""ONE command to run the complete monitoring + retraining workflow:

    python -m src.run_workflow
"""
import sys

from src import create_new_data, evaluate_model, quality_gate, retrain_model, validate_data


def step(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    step("STEP 1 - Create new incoming data batch")
    create_new_data.main()

    step("STEP 2 - Validate new data (Task A)")
    if not validate_data.main():
        print("\nValidation failed. Stopping the workflow.")
        sys.exit(1)

    step("STEP 3 - Evaluate existing champion model (Task B)")
    evaluate_model.main()

    step("STEP 4 - Retrain candidate model (Task C)")
    retrain_model.main()

    step("STEP 5 - Quality gate (Task D)")
    quality_gate.main()

    print("\nWorkflow finished. Reports are in the 'reports' folder.")


if __name__ == "__main__":
    main()
