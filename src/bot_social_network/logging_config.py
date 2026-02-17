import os
import logging
import datetime
from pythonjsonlogger import jsonlogger

def setup_logging():
    """
    Sets up a unique, timestamped directory for each simulation run,
    containing a JSONL log file and a directory for audio output.
    """
    base_log_dir = "logs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_log_dir, f"sim_{timestamp}")
    audio_dir = os.path.join(run_dir, "audio")
    
    os.makedirs(audio_dir, exist_ok=True)

    log_filename = os.path.join(run_dir, "simulation.jsonl")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler for the new log file
    log_handler = logging.FileHandler(log_filename)

    # Create a JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    log_handler.setFormatter(formatter)

    # Add the handler to the root logger
    logger.addHandler(log_handler)

    logging.info("Structured logging initialized.", extra={'event': 'system.init', 'run_dir': run_dir})
    return run_dir
