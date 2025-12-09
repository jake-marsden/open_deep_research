"""Tier classification logger for tracking adaptive model selection statistics."""

import os
from typing import Dict
from threading import Lock


class TierLogger:
    """Thread-safe logger for tracking tier classification counts."""
    
    def __init__(self, log_file: str = "logs.txt"):
        """Initialize the tier logger.
        
        Args:
            log_file: Path to the log file relative to the module directory.
        """
        # Get the directory of this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(module_dir, log_file)
        self._lock = Lock()
    
    def log_tier_classification(self, tier: str) -> None:
        """Log a tier classification and update the counts file.
        
        Args:
            tier: The tier classification ('low', 'mid', or 'high')
        """
        with self._lock:
            # Read current counts
            counts = self._read_counts()
            
            # Update count for the tier
            tier_key = f"{tier.lower()}-tier"
            if tier_key in counts:
                counts[tier_key] += 1
            
            # Write updated counts
            self._write_counts(counts)
    
    def _read_counts(self) -> Dict[str, int]:
        """Read current tier counts from the log file.
        
        Returns:
            Dictionary with tier counts (default: all zeros)
        """
        counts = {
            "low-tier": 0,
            "mid-tier": 0,
            "high-tier": 0
        }
        
        if not os.path.exists(self.log_file):
            return counts
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
            # Parse the counts from the file
            for line in lines:
                line = line.strip()
                if line.startswith("Low-tier:"):
                    counts["low-tier"] = int(line.split(":")[1].strip())
                elif line.startswith("Mid-tier:"):
                    counts["mid-tier"] = int(line.split(":")[1].strip())
                elif line.startswith("High-tier:"):
                    counts["high-tier"] = int(line.split(":")[1].strip())
        
        except Exception as e:
            # If parsing fails, return default counts
            pass
        
        return counts
    
    def _write_counts(self, counts: Dict[str, int]) -> None:
        """Write tier counts to the log file in the specified format.
        
        Args:
            counts: Dictionary with tier counts
        """
        low = counts.get("low-tier", 0)
        mid = counts.get("mid-tier", 0)
        high = counts.get("high-tier", 0)
        total = low + mid + high
        
        # Calculate percentages
        low_pct = (low / total * 100) if total > 0 else 0.0
        mid_pct = (mid / total * 100) if total > 0 else 0.0
        high_pct = (high / total * 100) if total > 0 else 0.0
        
        # Format the output
        output = f"""Adaptive Model Selection - Tier Classification Counts
=======================================================
Low-tier:  {low}
Mid-tier:  {mid}
High-tier: {high}
Total:     {total}

Distribution:
  Low:  {low_pct:.1f}%
  Mid:  {mid_pct:.1f}%
  High: {high_pct:.1f}%
"""
        
        # Write to file
        with open(self.log_file, 'w') as f:
            f.write(output)
    
    def reset_counts(self) -> None:
        """Reset all tier counts to zero."""
        with self._lock:
            counts = {
                "low-tier": 0,
                "mid-tier": 0,
                "high-tier": 0
            }
            self._write_counts(counts)


# Create a global singleton instance
_tier_logger_instance = None


def get_tier_logger(log_file: str = "logs.txt") -> TierLogger:
    """Get the global tier logger instance.
    
    Args:
        log_file: Path to the log file relative to the module directory.
        
    Returns:
        The global TierLogger instance
    """
    global _tier_logger_instance
    if _tier_logger_instance is None:
        _tier_logger_instance = TierLogger(log_file)
    return _tier_logger_instance

