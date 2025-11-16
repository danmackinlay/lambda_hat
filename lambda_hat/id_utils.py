"""Stable content-addressed ID generation utilities.

Provides simple hashing functions for creating reproducible identifiers
from configuration dictionaries. Reuses core logic from workflow_utils.py.
"""

import hashlib
import json


def stable_hash(obj, hash_len=12, algorithm="sha256"):
    """Compute stable content-addressed hash from Python object.

    Args:
        obj: Python object (dict, list, str, etc.) to hash
        hash_len: Number of hex characters to return (default: 12)
        algorithm: Hash algorithm ("sha256", "sha1", "blake2b") (default: "sha256")

    Returns:
        str: Hex string of length hash_len

    Examples:
        >>> stable_hash({"a": 1, "b": 2})
        '2cf24dba5fb0'
        >>> stable_hash({"b": 2, "a": 1})  # Same hash (key order normalized)
        '2cf24dba5fb0'
    """
    blob = json.dumps(obj, sort_keys=True)

    if algorithm == "sha256":
        return hashlib.sha256(blob.encode()).hexdigest()[:hash_len]
    elif algorithm == "sha1":
        return hashlib.sha1(blob.encode()).hexdigest()[:hash_len]
    elif algorithm == "blake2b":
        return hashlib.blake2b(blob.encode(), digest_size=hash_len // 2).hexdigest()[:hash_len]
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")


def problem_id(problem_spec):
    """Generate problem ID from problem specification.

    Args:
        problem_spec: Dict with problem configuration (model, data, teacher, seed)

    Returns:
        str: Problem ID with format 'p_<12-char-hash>'
    """
    return "p_" + stable_hash(problem_spec, hash_len=12)


def trial_id(trial_spec):
    """Generate trial ID from trial specification.

    Args:
        trial_spec: Dict with trial configuration (problem, method, hyperparams, seed)

    Returns:
        str: Trial ID with format 'r_<12-char-hash>'
    """
    return "r_" + stable_hash(trial_spec, hash_len=12)
