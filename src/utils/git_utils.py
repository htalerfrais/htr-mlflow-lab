import subprocess
import logging

logger = logging.getLogger(__name__)


def get_current_git_commit() -> str:
    """Get the current git commit hash.
    
    Returns:
        Git commit hash as string
        
    Raises:
        Exception: If git command fails
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()
        logger.info(f"Current git commit: {commit_hash}")
        return commit_hash
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get git commit: {e}")
        raise Exception(f"Failed to get git commit: {e}")
    except FileNotFoundError:
        logger.error("Git not found in PATH")
        raise Exception("Git not found in PATH")
