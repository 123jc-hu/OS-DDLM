import time

from training.trainer import OptimizedExperimentRunner
from utils.config import Logger, build_config
from utils.run_manifest import create_manifest, finalize_manifest


def run_experiment(config):
    """Run one configured experiment without changing training behavior."""
    start_time = time.time()
    logger = Logger(config.get("log_name", config["model"])).get_logger()
    logger.info(f'Starting training for model: {config["model"]}')

    experiment = OptimizedExperimentRunner(config, logger)
    logger.info(f"\n================ Running Dataset: {config['dataset']} ================\n")

    manifest_path = create_manifest(config, experiment.exp_dir)
    try:
        experiment.run_experiment()
    except Exception as error:
        finalize_manifest(manifest_path, experiment.exp_dir, experiment.run_provenance, "failed", str(error))
        raise
    finalize_manifest(manifest_path, experiment.exp_dir, experiment.run_provenance, "completed")

    logger.info("Finished training and testing!")
    logger.info(f"Selected K: {config.get('selector_K', 'N/A')}")
    logger.info(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes \n")


def main(config):
    """Backward-compatible alias for older imports."""
    run_experiment(config)


if __name__ == "__main__":
    config = build_config("DDLM-Extractor")  # Default paper-facing backbone name
    run_experiment(config)
