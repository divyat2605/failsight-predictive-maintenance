"""
Pipeline Scheduler: APScheduler-based daily pipeline runner
"""
import os
import sys
import subprocess
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_scheduler.log'),
        logging.StreamHandler()
    ]
)

PIPELINE_STEPS = [
    ("pipelines/ingest.py", "Data ingestion"),
    ("pipelines/features.py", "Feature engineering"),
    ("models/train_rul.py", "Model training"),
    ("agent/build_vectorstore.py", "Vector store rebuild")
]

def run_pipeline_step(script_path: str, description: str) -> bool:
    """Run a single pipeline step and return success status."""
    start_time = datetime.now()
    logging.info(f"Starting {description}...")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.returncode == 0:
            logging.info(f"SUCCESS: {description} completed in {duration:.1f}s")
            return True
        else:
            logging.error(f"FAILED: {description} with return code {result.returncode}")
            logging.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logging.error(f"✗ {description} timed out after 1 hour")
        return False
    except Exception as e:
        logging.error(f"✗ {description} failed with exception: {str(e)}")
        return False

def run_daily_pipeline():
    """Run the complete daily pipeline."""
    logging.info("=== Starting Daily Pipeline Run ===")
    start_time = datetime.now()

    results = []
    for script_path, description in PIPELINE_STEPS:
        success = run_pipeline_step(script_path, description)
        results.append((description, success))

        if not success:
            logging.warning(f"Pipeline stopped due to failure in {description}")
            break

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Summary
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    logging.info("=== Pipeline Run Summary ===")
    logging.info(f"Total time: {total_duration:.1f}s")
    logging.info(f"Steps passed: {passed}")
    logging.info(f"Steps failed: {failed}")

    for desc, success in results:
        status = "PASS" if success else "FAIL"
        logging.info(f"  {desc}: {status}")

    if failed > 0:
        logging.warning("Pipeline completed with failures")
    else:
        logging.info("Pipeline completed successfully")

def main():
    """Main scheduler function."""
    logging.info("Starting FailSight Pipeline Scheduler")
    logging.info("Scheduled to run daily at midnight (00:00)")

    scheduler = BlockingScheduler()

    # Schedule daily at midnight
    trigger = CronTrigger(hour=0, minute=0)
    scheduler.add_job(run_daily_pipeline, trigger=trigger, id='daily_pipeline')

    logging.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
        scheduler.shutdown()

if __name__ == "__main__":
    main()