"""
Semantic hash generation for code comparison.

Generates fingerprints of normalized code to detect renamed/moved functions.
"""

import ast
import hashlib
from typing import Dict, Tuple, Union

from codediff.parser import ClassInfo, FunctionInfo, NormalizationVisitor


class SemanticHasher:
    """Generate semantic hashes for code elements."""

    @staticmethod
    def hash_function(func_info: FunctionInfo, normalize_imports: bool = False) -> str:
        """Generate semantic hash for a function.

        Args:
            func_info: FunctionInfo object to hash
            normalize_imports: If True, replace mmcv with visdet.cv and mmengine with visdet.engine

        Returns:
            Hex string of semantic hash
        """
        # Normalize the source code for hashing
        normalized = NormalizationVisitor.normalize_source(func_info.source_code, normalize_imports=normalize_imports)

        # Try to parse and normalize the AST for more robust hashing
        try:
            tree = ast.parse(normalized)
            # Remove docstrings from AST
            transformer = NormalizationVisitor()
            tree = transformer.visit(tree)
            # Unparse to canonical form
            canonical = ast.unparse(tree)
        except Exception:
            # Fall back to source normalization
            canonical = normalized

        # Generate hash
        hasher = hashlib.sha256()
        hasher.update(canonical.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def hash_class(class_info: ClassInfo) -> str:
        """Generate semantic hash for a class.

        Hashes the class definition including method signatures.

        Args:
            class_info: ClassInfo object to hash

        Returns:
            Hex string of semantic hash
        """
        # Combine hashes of all methods
        hasher = hashlib.sha256()

        # Hash the class name and bases
        hasher.update(class_info.name.encode("utf-8"))
        for base in sorted(class_info.bases):
            hasher.update(base.encode("utf-8"))

        # Hash method signatures (not implementations, as they may vary)
        for method_name in sorted(class_info.methods.keys()):
            method = class_info.methods[method_name]
            hasher.update(method.name.encode("utf-8"))
            hasher.update(method.signature.encode("utf-8"))

        return hasher.hexdigest()

    @staticmethod
    def calculate_similarity(hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes.

        Uses Hamming distance to measure how different the hashes are.

        Args:
            hash1: First hash string
            hash2: Second hash string

        Returns:
            Float between 0 (identical) and 1 (completely different)
        """
        if hash1 == hash2:
            return 0.0

        if len(hash1) != len(hash2):
            return 1.0

        # Count differing bits
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        max_distance = len(hash1)

        return distance / max_distance


class HashMatcher:
    """Match unmatched code items by semantic similarity."""

    SIMILARITY_THRESHOLD: float = 0.15  # Items within 15% hash distance are considered matches

    @staticmethod
    def find_matches(
        source_items: Dict[str, Union[FunctionInfo, ClassInfo]],
        target_items: Dict[str, Union[FunctionInfo, ClassInfo]],
        normalize_imports: bool = False,
    ) -> Dict[str, Tuple[str, float]]:
        """Find potential name-based matches between source and target items.

        Args:
            source_items: Source code items keyed by name
            target_items: Target code items keyed by name
            normalize_imports: If True, replace mmcv with visdet.cv and mmengine with visdet.engine

        Returns:
            Dict mapping source names to (target_name, similarity) tuples
        """
        matches: Dict[str, Tuple[str, float]] = {}

        # Generate hashes for all items
        source_hashes = {}
        for name, item in source_items.items():
            if isinstance(item, FunctionInfo):
                source_hashes[name] = SemanticHasher.hash_function(item, normalize_imports=normalize_imports)
            else:
                source_hashes[name] = SemanticHasher.hash_class(item)

        target_hashes = {}
        for name, item in target_items.items():
            if isinstance(item, FunctionInfo):
                target_hashes[name] = SemanticHasher.hash_function(item, normalize_imports=normalize_imports)
            else:
                target_hashes[name] = SemanticHasher.hash_class(item)

        # Find best matches for each source item
        for src_name, src_hash in source_hashes.items():
            best_match = None
            best_similarity = 1.0

            for tgt_name, tgt_hash in target_hashes.items():
                similarity = SemanticHasher.calculate_similarity(src_hash, tgt_hash)

                # Track the best match
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_match = tgt_name

            # Only record if above threshold
            if best_match and best_similarity < HashMatcher.SIMILARITY_THRESHOLD:
                matches[src_name] = (best_match, best_similarity)

        return matches
