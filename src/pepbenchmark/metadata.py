import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_DIR = BASE_DIR / "PepBenchData" / "PepBenchData-50"
DATASET_DIR = os.environ.get("PEPBENCHMARK_DATASET_DIR", DEFAULT_DATASET_DIR)

class DatasetGroup(str, Enum):
    ADME = "ADME"
    AMP = "AMP"
    METABOLIC = "Metabolic"
    ONCOLOGY = "Oncology"
    OTHERS = "Others"
    PEPPI = "PepPI"
    PHYSCHEM = "PhysChem"
    TOX = "Tox"


class DatasetType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    group: DatasetGroup
    type: DatasetType
    is_nature: bool
    path: str | None = None
    neg_path: str | None = None


DATASET_MAP: dict[str, DatasetMeta] = {
    # ======================== ADME ========================
    "bbp": DatasetMeta(
        group=DatasetGroup.ADME,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "cpp": DatasetMeta(
        group=DatasetGroup.ADME,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "nc-cpp_pampa": DatasetMeta(
        group=DatasetGroup.ADME,
        type=DatasetType.REGRESSION,
        is_nature=False,
    ),
    "nonfouling": DatasetMeta(
        group=DatasetGroup.ADME,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    # ======================== AMP ========================
    "antibacterial": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "antifungal": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "antimicrobial": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "antimicrobial_E_coli_mic": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.REGRESSION,
        is_nature=True,
    ),
    "antimicrobial_P_aeruginosa_mic": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.REGRESSION,
        is_nature=True,
    ),
    "antimicrobial_S_aureus_mic": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.REGRESSION,
        is_nature=True,
    ),
    "antiparasitic": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "antiviral": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "nc-antibacterial": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=False,
        neg_path=None,
    ),
    "nc-antifungal": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=False,
        neg_path=None,
    ),
    "nc-antimicrobial": DatasetMeta(
        group=DatasetGroup.AMP,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=False,
        neg_path=None,
    ),

    # ======================== Metabolic ========================
    "ace_inhibitory": DatasetMeta(
        group=DatasetGroup.METABOLIC,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "ace_inhibitory_ic50": DatasetMeta(
        group=DatasetGroup.METABOLIC,
        type=DatasetType.REGRESSION,
        is_nature=True,
    ),
    "antidiabetic": DatasetMeta(
        group=DatasetGroup.METABOLIC,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "dppiv_inhibitors": DatasetMeta(
        group=DatasetGroup.METABOLIC,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),

    # ======================== Oncology ========================
    "anticancer": DatasetMeta(
        group=DatasetGroup.ONCOLOGY,  # 修复原代码缺失 group 的问题
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "ttca": DatasetMeta(
        group=DatasetGroup.ONCOLOGY,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),

    # ======================== Others ========================
    "antiaging": DatasetMeta(
        group=DatasetGroup.OTHERS,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "antiinflamatory": DatasetMeta(
        group=DatasetGroup.OTHERS,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "antioxidant": DatasetMeta(
        group=DatasetGroup.OTHERS,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "neuropeptide": DatasetMeta(
        group=DatasetGroup.OTHERS,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),
    "quorum_sensing": DatasetMeta(
        group=DatasetGroup.OTHERS,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
        neg_path=None,
    ),

    # ======================== PepPI ========================
    "PpI": DatasetMeta(
        group=DatasetGroup.PEPPI,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "PpI_ba": DatasetMeta(
        group=DatasetGroup.PEPPI,
        type=DatasetType.REGRESSION,
        is_nature=True,
    ),
    "nc-PpI_ba": DatasetMeta(
        group=DatasetGroup.PEPPI,
        type=DatasetType.REGRESSION,
        is_nature=False,
    ),



    # ======================== Tox ========================
    "allergen": DatasetMeta(
        group=DatasetGroup.TOX,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "hemolytic": DatasetMeta(
        group=DatasetGroup.TOX,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "hemolytic_hc50": DatasetMeta(
        group=DatasetGroup.TOX,
        type=DatasetType.REGRESSION,
        is_nature=True,
    ),
    # "hemolytic_strict": DatasetMeta(
    #     group=DatasetGroup.TOX,
    #     type=DatasetType.BINARY_CLASSIFICATION,
    #     is_nature=True,
    # ),
    "neurotoxin": DatasetMeta(
        group=DatasetGroup.TOX,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "toxicity": DatasetMeta(
        group=DatasetGroup.TOX,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=True,
    ),
    "nc-hemolytic": DatasetMeta(
        group=DatasetGroup.TOX,
        type=DatasetType.BINARY_CLASSIFICATION,
        is_nature=False,
    ),
}


def _normalize_group(group: str | DatasetGroup | None) -> DatasetGroup | None:
    if group is None:
        return None
    if isinstance(group, DatasetGroup):
        return group
    return DatasetGroup(group)


def _normalize_type(dataset_type: str | DatasetType | None) -> DatasetType | None:
    if dataset_type is None:
        return None
    if isinstance(dataset_type, DatasetType):
        return dataset_type
    return DatasetType(dataset_type)


def get_dataset(name: str) -> DatasetMeta:
    """根据数据集名称获取元信息，不存在时抛出清晰异常。"""
    try:
        return DATASET_MAP[name]
    except KeyError as e:
        raise KeyError(f"Unknown dataset: {name!r}") from e


def has_dataset(name: str) -> bool:
    return name in DATASET_MAP


def list_dataset_names() -> list[str]:
    return list(DATASET_MAP.keys())


@lru_cache(maxsize=None)
def get_datasets_by_group(group: str | DatasetGroup) -> dict[str, DatasetMeta]:
    group = _normalize_group(group)
    assert group is not None
    return {
        name: meta
        for name, meta in DATASET_MAP.items()
        if meta.group == group
    }


@lru_cache(maxsize=None)
def get_datasets_by_type(dataset_type: str | DatasetType) -> dict[str, DatasetMeta]:
    dataset_type = _normalize_type(dataset_type)
    assert dataset_type is not None
    return {
        name: meta
        for name, meta in DATASET_MAP.items()
        if meta.type == dataset_type
    }


@lru_cache(maxsize=None)
def get_datasets_by_nature(is_nature: bool) -> dict[str, DatasetMeta]:
    return {
        name: meta
        for name, meta in DATASET_MAP.items()
        if meta.is_nature == is_nature
    }


def list_datasets(
    *,
    group: str | DatasetGroup | None = None,
    dataset_type: str | DatasetType | None = None,
    is_nature: bool | None = None,
    has_path: bool | None = None,
    has_neg_path: bool | None = None,
) -> dict[str, DatasetMeta]:
    """
        list_datasets(group=DatasetGroup.AMP)
        list_datasets(dataset_type=DatasetType.REGRESSION, is_nature=True)
        list_datasets(group=DatasetGroup.TOX, dataset_type=DatasetType.BINARY_CLASSIFICATION)
    """
    group = _normalize_group(group)
    dataset_type = _normalize_type(dataset_type)

    result: dict[str, DatasetMeta] = {}
    for name, meta in DATASET_MAP.items():
        if group is not None and meta.group != group:
            continue
        if dataset_type is not None and meta.type != dataset_type:
            continue
        if is_nature is not None and meta.is_nature != is_nature:
            continue
        if has_path is not None and ((meta.path is not None) != has_path):
            continue
        if has_neg_path is not None and ((meta.neg_path is not None) != has_neg_path):
            continue
        result[name] = meta
    return result




