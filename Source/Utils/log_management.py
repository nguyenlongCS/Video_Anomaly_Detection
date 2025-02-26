import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil


def cleanup_old_logs(max_age_days=30):
    """
    Clean up log files older than specified days

    Args:
        max_age_days: Maximum age of log files in days
    """
    log_dir = Path('logs')
    current_time = datetime.utcnow()

    # Process each subdirectory
    for subdir in ['training', 'inference', 'preprocessing']:
        dir_path = log_dir / subdir
        if not dir_path.exists():
            continue

        # Check each log file
        for log_file in dir_path.glob('*.log'):
            # Get file creation time from filename
            try:
                file_date = datetime.strptime(log_file.stem.split('_')[1], '%Y%m%d')
                if current_time - file_date > timedelta(days=max_age_days):
                    os.remove(log_file)
                    print(f"Removed old log file: {log_file}")
            except Exception as e:
                print(f"Error processing {log_file}: {str(e)}")


def archive_logs(archive_dir='log_archives'):
    """
    Archive current logs

    Args:
        archive_dir: Directory for storing archives
    """
    log_dir = Path('logs')
    archive_path = Path(archive_dir)
    archive_path.mkdir(exist_ok=True)

    # Create archive name with timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    archive_name = f'logs_archive_{timestamp}.zip'

    # Create archive
    shutil.make_archive(
        str(archive_path / archive_name.replace('.zip', '')),
        'zip',
        str(log_dir)
    )
    print(f"Logs archived to: {archive_path / archive_name}")


if __name__ == '__main__':
    # Clean up old logs
    cleanup_old_logs()

    # Archive current logs
    archive_logs()