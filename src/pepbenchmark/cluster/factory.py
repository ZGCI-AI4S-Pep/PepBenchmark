# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified clustering factory for creating different clustering algorithms.

This module provides a centralized factory for creating clustering instances
with consistent interfaces and configuration management.
"""

from dataclasses import fields
from typing import Any, Dict, Optional, Type
from enum import Enum

from pepbenchmark.cluster.interfaces import AbstractClusterer, ClusterConfig
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class ClusteringMethod(Enum):
    """Supported clustering methods."""
    CDHIT = "cdhit"
    MMSEQS2 = "mmseqs2"
    MOTIF = "motif"
    KMER = "kmer"
    SIMILARITY = "similarity"


class ClusterFactory:
    """
    Factory class for creating clustering instances.
    
    This factory provides a unified interface for creating different types of
    clustering algorithms with consistent configuration and validation.
    """
    
    _registry: Dict[str, Type[AbstractClusterer]] = {}
    _config_registry: Dict[str, Type[ClusterConfig]] = {}

    @staticmethod
    def _normalize_config_kwargs(method: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize backward-compatible parameter aliases before config creation."""
        clean_kwargs = dict(kwargs)

        if method == "mmseqs2" and "identity" in clean_kwargs:
            clean_kwargs["min_seq_id"] = clean_kwargs.pop("identity")

        if "threshold" in clean_kwargs:
            threshold_val = clean_kwargs.pop("threshold")
            if method == "similarity":
                clean_kwargs.setdefault("similarity_threshold", threshold_val)
            elif method == "mmseqs2":
                clean_kwargs.setdefault("min_seq_id", threshold_val)
            elif method == "cdhit":
                clean_kwargs.setdefault("c", threshold_val)

        if method == "cdhit" and "identity" in clean_kwargs:
            clean_kwargs.setdefault("c", clean_kwargs.pop("identity"))

        return clean_kwargs

    @classmethod
    def _extract_config_kwargs(
        cls,
        method: str,
        config_class: Type[ClusterConfig],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Keep only config fields so runtime arguments are not passed to config dataclasses."""
        aliased_kwargs = cls._normalize_config_kwargs(method, kwargs)
        valid_field_names = {field.name for field in fields(config_class)}
        return {
            key: value
            for key, value in aliased_kwargs.items()
            if key in valid_field_names
        }
    
    @classmethod
    def register_clusterer(
        cls,
        method: str,
        clusterer_class: Type[AbstractClusterer],
        config_class: Type[ClusterConfig]
    ) -> None:
        """
        Register a new clustering method.
        
        Args:
            method: Method name (e.g., "cdhit", "mmseqs2")
            clusterer_class: Clusterer implementation class
            config_class: Configuration class for the clusterer
        """
        cls._registry[method] = clusterer_class
        cls._config_registry[method] = config_class
        logger.info(f"Registered clustering method: {method}")
    
    @classmethod
    def create_clusterer(
        cls,
        method: str,
        config: Optional[ClusterConfig] = None,
        **kwargs: Any
    ) -> AbstractClusterer:
        """
        Create a clusterer instance.
        
        Args:
            method: Clustering method name
            config: Optional configuration object
            **kwargs: Configuration parameters (used if config is None)
            
        Returns:
            Configured clusterer instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in cls._registry:
            available_methods = list(cls._registry.keys())
            raise ValueError(f"Unsupported clustering method: {method}. "
                           f"Available methods: {available_methods}")
        
        impl_class = cls._registry[method]
        
        # Create config if not provided
        if config is None:
            cfg_class = cls._config_registry[method]

            cfg_kwargs = cls._extract_config_kwargs(method, cfg_class, kwargs)
            config = cfg_class(**cfg_kwargs)
        
        return impl_class(config)
    
    @classmethod
    def list_methods(cls) -> list[str]:
        """List all registered clustering methods."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_config_class(cls, method: str) -> Type[ClusterConfig]:
        """Get the configuration class for a clustering method."""
        if method not in cls._config_registry:
            raise ValueError(f"Unknown clustering method: {method}")
        return cls._config_registry[method]


# Convenience functions
def create_clusterer(method: str, **kwargs) -> AbstractClusterer:
    """
    Convenience function to create a clusterer.
    
    Args:
        method: Clustering method name
        **kwargs: Configuration parameters
        
    Returns:
        Configured clusterer instance
    """
    return ClusterFactory.create_clusterer(method, **kwargs)


def create_cdhit_clusterer(**kwargs) -> AbstractClusterer:
    """Create CD-HIT clusterer with default parameters."""
    return create_clusterer("cdhit", **kwargs)


def create_mmseqs2_clusterer(**kwargs) -> AbstractClusterer:
    """Create MMseqs2 clusterer with default parameters."""
    return create_clusterer("mmseqs2", **kwargs)


def create_kmer_clusterer(**kwargs) -> AbstractClusterer:
    """Create kmer-based clusterer with default parameters."""
    return create_clusterer("kmer", **kwargs)


def create_motif_clusterer(**kwargs) -> AbstractClusterer:
    """Backward-compatible alias for the kmer-based clusterer."""
    return create_kmer_clusterer(**kwargs)


def create_similarity_clusterer(**kwargs) -> AbstractClusterer:
    """Create similarity-based clusterer with default parameters."""
    return create_clusterer("similarity", **kwargs)


def list_available_methods() -> list[str]:
    """List all available clustering methods."""
    return ClusterFactory.list_methods()
