# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Compatibility layer for NumPy 2.0+ with sinter."""

import numpy as np

# Check if we're using NumPy 2.0+
_NUMPY_2_PLUS = int(np.__version__.split(".")[0]) >= 2


def _apply_numpy_compat_patch():
    """Apply compatibility patch for NumPy 2.0+ with sinter.
    
    NumPy 2.0+ changed integer type behavior: numpy.int64 is no longer
    considered an instance of Python's int. This breaks sinter's internal
    validation. This patch ensures numpy integers are converted to Python ints.
    """
    if not _NUMPY_2_PLUS:
        return  # No patch needed for NumPy < 2.0
    
    try:
        from sinter._data import _anon_task_stats
        
        # Patch AnonTaskStats to handle numpy integers
        # The error occurs in sinter._data._anon_task_stats.AnonTaskStats.__post_init__
        # We need to convert numpy integers before the validation happens
        if hasattr(_anon_task_stats, "AnonTaskStats"):
            AnonTaskStats = _anon_task_stats.AnonTaskStats
            
            # Store original __post_init__
            original_post_init = getattr(AnonTaskStats, "__post_init__", None)
            
            def patched_post_init(self):
                """Patched __post_init__ that converts numpy integers to Python ints."""
                # Convert numpy integers to Python ints before validation
                # Use object.__setattr__ to bypass frozen dataclass restriction if needed
                if hasattr(self, "errors") and isinstance(self.errors, (np.integer, np.int_)):
                    object.__setattr__(self, "errors", int(self.errors))
                if hasattr(self, "shots") and isinstance(self.shots, (np.integer, np.int_)):
                    object.__setattr__(self, "shots", int(self.shots))
                if hasattr(self, "discards") and isinstance(self.discards, (np.integer, np.int_)):
                    object.__setattr__(self, "discards", int(self.discards))
                
                # Call original __post_init__ if it exists (this will do the validation)
                if original_post_init is not None:
                    original_post_init(self)
            
            # Apply the patch
            AnonTaskStats.__post_init__ = patched_post_init
                    
    except (ImportError, AttributeError):
        # Sinter not available or structure changed, skip patching
        pass


# Automatically apply the patch when this module is imported
_apply_numpy_compat_patch()

