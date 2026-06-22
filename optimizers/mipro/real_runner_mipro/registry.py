"""Central registry for real-runner MIPRO pairs.

A pair is identified by `(dataset, topology)`, for example
`("bfcl", "sequential_crewai")`. New datasets should add their adapter map
here after implementing the dataset loader, metric, and adapters.
"""
from __future__ import annotations

from importlib import import_module

from real_runner_mipro.protocol import RealRunnerAdapter


AdapterClass = type[RealRunnerAdapter]


DATASET_ADAPTERS: dict[str, dict[str, str]] = {
    "bfcl": {
        "single": "real_runner_mipro.adapters.single_bfcl:SingleBFCLAdapter",
        "independent": "real_runner_mipro.adapters.independent_bfcl:IndependentBFCLAdapter",
        "decentralized": "real_runner_mipro.adapters.decentralized_bfcl:DecentralizedBFCLAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.decentralized_openai_bfcl:DecentralizedOpenAIBFCLAdapter",
        "sequential": "real_runner_mipro.adapters.sequential_bfcl:SequentialBFCLAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.sequential_crewai_bfcl:SequentialCrewAIBFCLAdapter",
        "centralized": "real_runner_mipro.adapters.centralized_bfcl:CentralizedBFCLAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.centralized_autogen_bfcl:CentralizedAutoGenBFCLAdapter",
    },
    "gpqa": {
        "single": "real_runner_mipro.adapters.single_gpqa:SingleGPQAAdapter",
        "independent": "real_runner_mipro.adapters.independent_gpqa:IndependentGPQAAdapter",
        "decentralized": "real_runner_mipro.adapters.module_gpqa:DecentralizedGPQAAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_gpqa:DecentralizedOpenAIGPQAAdapter",
        "sequential": "real_runner_mipro.adapters.module_gpqa:SequentialGPQAAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_gpqa:SequentialCrewAIGPQAAdapter",
        "centralized": "real_runner_mipro.adapters.module_gpqa:CentralizedGPQAAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_gpqa:CentralizedAutoGenGPQAAdapter",
    },
    "hotpotqa": {
        "single": "real_runner_mipro.adapters.module_hotpotqa:SingleHotpotQAAdapter",
        "independent": "real_runner_mipro.adapters.module_hotpotqa:IndependentHotpotQAAdapter",
        "decentralized": "real_runner_mipro.adapters.module_hotpotqa:DecentralizedHotpotQAAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_hotpotqa:DecentralizedOpenAIHotpotQAAdapter",
        "sequential": "real_runner_mipro.adapters.module_hotpotqa:SequentialHotpotQAAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_hotpotqa:SequentialCrewAIHotpotQAAdapter",
        "centralized": "real_runner_mipro.adapters.module_hotpotqa:CentralizedHotpotQAAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_hotpotqa:CentralizedAutoGenHotpotQAAdapter",
    },
    "math": {
        "single": "real_runner_mipro.adapters.module_math:SingleMATHAdapter",
        "independent": "real_runner_mipro.adapters.module_math:IndependentMATHAdapter",
        "decentralized": "real_runner_mipro.adapters.module_math:DecentralizedMATHAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_math:DecentralizedOpenAIMATHAdapter",
        "sequential": "real_runner_mipro.adapters.module_math:SequentialMATHAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_math:SequentialCrewAIMATHAdapter",
        "centralized": "real_runner_mipro.adapters.module_math:CentralizedMATHAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_math:CentralizedAutoGenMATHAdapter",
    },
    "apps": {
        "single": "real_runner_mipro.adapters.module_apps:SingleAPPSAdapter",
        "independent": "real_runner_mipro.adapters.module_apps:IndependentAPPSAdapter",
        "decentralized": "real_runner_mipro.adapters.module_apps:DecentralizedAPPSAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_apps:DecentralizedOpenAIAPPSAdapter",
        "sequential": "real_runner_mipro.adapters.module_apps:SequentialAPPSAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_apps:SequentialCrewAIAPPSAdapter",
        "centralized": "real_runner_mipro.adapters.module_apps:CentralizedAPPSAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_apps:CentralizedAutoGenAPPSAdapter",
    },
    "lcb": {
        "single": "real_runner_mipro.adapters.module_lcb:SingleLCBAdapter",
        "independent": "real_runner_mipro.adapters.module_lcb:IndependentLCBAdapter",
        "decentralized": "real_runner_mipro.adapters.module_lcb:DecentralizedLCBAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_lcb:DecentralizedOpenAILCBAdapter",
        "sequential": "real_runner_mipro.adapters.module_lcb:SequentialLCBAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_lcb:SequentialCrewAILCBAdapter",
        "centralized": "real_runner_mipro.adapters.module_lcb:CentralizedLCBAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_lcb:CentralizedAutoGenLCBAdapter",
    },
    "swe": {
        "single": "real_runner_mipro.adapters.module_swe:SingleSWEAdapter",
        "independent": "real_runner_mipro.adapters.module_swe:IndependentSWEAdapter",
        "decentralized": "real_runner_mipro.adapters.module_swe:DecentralizedSWEAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_swe:DecentralizedOpenAISWEAdapter",
        "sequential": "real_runner_mipro.adapters.module_swe:SequentialSWEAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_swe:SequentialCrewAISWEAdapter",
        "centralized": "real_runner_mipro.adapters.module_swe:CentralizedSWEAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_swe:CentralizedAutoGenSWEAdapter",
    },
    "apibank": {
        "single": "real_runner_mipro.adapters.module_apibank:SingleAPIBankAdapter",
        "independent": "real_runner_mipro.adapters.module_apibank:IndependentAPIBankAdapter",
        "decentralized": "real_runner_mipro.adapters.module_apibank:DecentralizedAPIBankAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_apibank:DecentralizedOpenAIAPIBankAdapter",
        "sequential": "real_runner_mipro.adapters.module_apibank:SequentialAPIBankAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_apibank:SequentialCrewAIAPIBankAdapter",
        "centralized": "real_runner_mipro.adapters.module_apibank:CentralizedAPIBankAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_apibank:CentralizedAutoGenAPIBankAdapter",
    },
    "toolhop": {
        "single": "real_runner_mipro.adapters.module_toolhop:SingleToolHopAdapter",
        "independent": "real_runner_mipro.adapters.module_toolhop:IndependentToolHopAdapter",
        "decentralized": "real_runner_mipro.adapters.module_toolhop:DecentralizedToolHopAdapter",
        "decentralized_openai": "real_runner_mipro.adapters.module_toolhop:DecentralizedOpenAIToolHopAdapter",
        "sequential": "real_runner_mipro.adapters.module_toolhop:SequentialToolHopAdapter",
        "sequential_crewai": "real_runner_mipro.adapters.module_toolhop:SequentialCrewAIToolHopAdapter",
        "centralized": "real_runner_mipro.adapters.module_toolhop:CentralizedToolHopAdapter",
        "centralized_autogen": "real_runner_mipro.adapters.module_toolhop:CentralizedAutoGenToolHopAdapter",
    },
}

_TEAMSIZES_DATASETS = ("hotpotqa", "lcb")
_TEAMSIZES_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
TEAM_SIZES = (2, 4, 8, 10)  # canonical team-size sweep — single source of truth
_COMMUNICATIONS_DATASETS = ("hotpotqa", "lcb")
_COMMUNICATIONS_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
_COMMUNICATIONS_FORMATS = ("freeform", "semi_structured", "structured_soft")


def _teamsizes_class_name(dataset: str, base_topology: str, team_size: int) -> str:
    dataset_prefix = {"hotpotqa": "HotpotQATeamSizes", "lcb": "LCBTeamSizes"}[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    return f"{dataset_prefix}{topo_prefix}R{team_size}Adapter"


def _communications_class_name(dataset: str, base_topology: str, fmt: str) -> str:
    dataset_prefix = {"hotpotqa": "HotpotQACommunications", "lcb": "LCBCommunications"}[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    fmt_prefix = "".join(part.title() for part in fmt.split("_"))
    return f"{dataset_prefix}{topo_prefix}{fmt_prefix}Adapter"


for _dataset in _TEAMSIZES_DATASETS:
    for _base_topology in _TEAMSIZES_BASE_TOPOLOGIES:
        for _team_size in TEAM_SIZES:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_r{_team_size}"] = (
                "real_runner_mipro.adapters.module_teamsizes:"
                f"{_teamsizes_class_name(_dataset, _base_topology, _team_size)}"
            )

for _dataset in _COMMUNICATIONS_DATASETS:
    for _base_topology in _COMMUNICATIONS_BASE_TOPOLOGIES:
        for _fmt in _COMMUNICATIONS_FORMATS:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_communications_{_fmt}"] = (
                "real_runner_mipro.adapters.module_communications:"
                f"{_communications_class_name(_dataset, _base_topology, _fmt)}"
            )


_TOOLUSE_DATASETS = ("apibank", "toolhop")
_TOOLUSE_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
_TOOLUSE_COMMUNICATIONS_FORMATS = ("freeform", "semi_structured", "structured_soft")
_TOOLUSE_PREFIXES = {"apibank": "APIBank", "toolhop": "ToolHop"}


def _tooluse_teamsizes_class_name(dataset: str, base_topology: str, team_size: int) -> str:
    prefix = _TOOLUSE_PREFIXES[dataset] + "TeamSizes"
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    return f"{prefix}{topo_prefix}R{team_size}Adapter"


def _tooluse_communications_class_name(dataset: str, base_topology: str, fmt: str) -> str:
    prefix = _TOOLUSE_PREFIXES[dataset] + "Communications"
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    fmt_prefix = "".join(part.title() for part in fmt.split("_"))
    return f"{prefix}{topo_prefix}{fmt_prefix}Adapter"


for _dataset in _TOOLUSE_DATASETS:
    for _base_topology in _TOOLUSE_BASE_TOPOLOGIES:
        for _team_size in TEAM_SIZES:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_r{_team_size}"] = (
                f"real_runner_mipro.adapters.module_{_dataset}:"
                f"{_tooluse_teamsizes_class_name(_dataset, _base_topology, _team_size)}"
            )
        for _fmt in _TOOLUSE_COMMUNICATIONS_FORMATS:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_communications_{_fmt}"] = (
                f"real_runner_mipro.adapters.module_{_dataset}:"
                f"{_tooluse_communications_class_name(_dataset, _base_topology, _fmt)}"
            )


def datasets() -> list[str]:
    """Return registered dataset names."""
    return sorted(DATASET_ADAPTERS)


def topologies(dataset: str) -> list[str]:
    """Return topology names for a dataset."""
    if dataset not in DATASET_ADAPTERS:
        raise KeyError(f"Unknown dataset {dataset!r}; choices={datasets()}")
    return sorted(DATASET_ADAPTERS[dataset])


def adapter_choices(dataset: str | None = None) -> list[str]:
    """Return CLI-friendly choices.

    For a specific dataset this returns bare topology names. Without a
    dataset, choices are returned as `dataset/topology`.
    """
    if dataset is not None:
        return topologies(dataset)
    choices: list[str] = []
    for ds in datasets():
        choices.extend(f"{ds}/{topology}" for topology in topologies(ds))
    return choices


def get_adapter_class(dataset: str, topology: str) -> AdapterClass:
    """Return the adapter class for a registered pair."""
    if dataset not in DATASET_ADAPTERS:
        raise KeyError(f"Unknown dataset {dataset!r}; choices={datasets()}")
    adapters = DATASET_ADAPTERS[dataset]
    if topology not in adapters:
        raise KeyError(
            f"Unknown topology {topology!r} for dataset {dataset!r}; "
            f"choices={sorted(adapters)}"
        )
    module_name, class_name = adapters[topology].split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, class_name)
