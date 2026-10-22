"""Central registry of real-runner pairs, shared by every optimizer.

A pair is identified by `(dataset, topology)`, for example
`("bfcl", "sequential_crewai")`. New datasets should add their adapter map
here after implementing the dataset loader, metric, and adapters.
"""

from __future__ import annotations

from importlib import import_module

from optimizers.bridge.adapter_protocol import RealRunnerAdapter

AdapterClass = type[RealRunnerAdapter]


DATASET_ADAPTERS: dict[str, dict[str, str]] = {
    "bfcl": {
        "single": "optimizers.bridge.adapters.single_bfcl:SingleBFCLAdapter",
        "independent": "optimizers.bridge.adapters.independent_bfcl:IndependentBFCLAdapter",
        "decentralized": "optimizers.bridge.adapters.decentralized_bfcl:DecentralizedBFCLAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.decentralized_openai_agents_bfcl:DecentralizedOpenAIAgentsBFCLAdapter",
        "sequential": "optimizers.bridge.adapters.sequential_bfcl:SequentialBFCLAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.sequential_crewai_bfcl:SequentialCrewAIBFCLAdapter",
        "centralized": "optimizers.bridge.adapters.centralized_bfcl:CentralizedBFCLAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.centralized_autogen_bfcl:CentralizedAutoGenBFCLAdapter",
    },
    "gpqa": {
        "single": "optimizers.bridge.adapters.single_gpqa:SingleGPQAAdapter",
        "independent": "optimizers.bridge.adapters.independent_gpqa:IndependentGPQAAdapter",
        "decentralized": "optimizers.bridge.adapters.module_gpqa:DecentralizedGPQAAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_gpqa:DecentralizedOpenAIAgentsGPQAAdapter",
        "sequential": "optimizers.bridge.adapters.module_gpqa:SequentialGPQAAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_gpqa:SequentialCrewAIGPQAAdapter",
        "centralized": "optimizers.bridge.adapters.module_gpqa:CentralizedGPQAAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_gpqa:CentralizedAutoGenGPQAAdapter",
    },
    "hotpotqa": {
        "single": "optimizers.bridge.adapters.module_hotpotqa:SingleHotpotQAAdapter",
        "independent": "optimizers.bridge.adapters.module_hotpotqa:IndependentHotpotQAAdapter",
        "decentralized": "optimizers.bridge.adapters.module_hotpotqa:DecentralizedHotpotQAAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_hotpotqa:DecentralizedOpenAIAgentsHotpotQAAdapter",
        "sequential": "optimizers.bridge.adapters.module_hotpotqa:SequentialHotpotQAAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_hotpotqa:SequentialCrewAIHotpotQAAdapter",
        "centralized": "optimizers.bridge.adapters.module_hotpotqa:CentralizedHotpotQAAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_hotpotqa:CentralizedAutoGenHotpotQAAdapter",
    },
    "math": {
        "single": "optimizers.bridge.adapters.module_math:SingleMATHAdapter",
        "independent": "optimizers.bridge.adapters.module_math:IndependentMATHAdapter",
        "decentralized": "optimizers.bridge.adapters.module_math:DecentralizedMATHAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_math:DecentralizedOpenAIAgentsMATHAdapter",
        "sequential": "optimizers.bridge.adapters.module_math:SequentialMATHAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_math:SequentialCrewAIMATHAdapter",
        "centralized": "optimizers.bridge.adapters.module_math:CentralizedMATHAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_math:CentralizedAutoGenMATHAdapter",
    },
    "apps": {
        "single": "optimizers.bridge.adapters.module_apps:SingleAPPSAdapter",
        "independent": "optimizers.bridge.adapters.module_apps:IndependentAPPSAdapter",
        "decentralized": "optimizers.bridge.adapters.module_apps:DecentralizedAPPSAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_apps:DecentralizedOpenAIAgentsAPPSAdapter",
        "sequential": "optimizers.bridge.adapters.module_apps:SequentialAPPSAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_apps:SequentialCrewAIAPPSAdapter",
        "centralized": "optimizers.bridge.adapters.module_apps:CentralizedAPPSAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_apps:CentralizedAutoGenAPPSAdapter",
    },
    "lcb": {
        "single": "optimizers.bridge.adapters.module_lcb:SingleLCBAdapter",
        "independent": "optimizers.bridge.adapters.module_lcb:IndependentLCBAdapter",
        "decentralized": "optimizers.bridge.adapters.module_lcb:DecentralizedLCBAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_lcb:DecentralizedOpenAIAgentsLCBAdapter",
        "sequential": "optimizers.bridge.adapters.module_lcb:SequentialLCBAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_lcb:SequentialCrewAILCBAdapter",
        "centralized": "optimizers.bridge.adapters.module_lcb:CentralizedLCBAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_lcb:CentralizedAutoGenLCBAdapter",
    },
    "swe": {
        "single": "optimizers.bridge.adapters.module_swe:SingleSWEAdapter",
        "independent": "optimizers.bridge.adapters.module_swe:IndependentSWEAdapter",
        "decentralized": "optimizers.bridge.adapters.module_swe:DecentralizedSWEAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_swe:DecentralizedOpenAIAgentsSWEAdapter",
        "sequential": "optimizers.bridge.adapters.module_swe:SequentialSWEAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_swe:SequentialCrewAISWEAdapter",
        "centralized": "optimizers.bridge.adapters.module_swe:CentralizedSWEAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_swe:CentralizedAutoGenSWEAdapter",
    },
    "apibank": {
        "single": "optimizers.bridge.adapters.module_apibank:SingleAPIBankAdapter",
        "independent": "optimizers.bridge.adapters.module_apibank:IndependentAPIBankAdapter",
        "decentralized": "optimizers.bridge.adapters.module_apibank:DecentralizedAPIBankAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_apibank:DecentralizedOpenAIAgentsAPIBankAdapter",
        "sequential": "optimizers.bridge.adapters.module_apibank:SequentialAPIBankAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_apibank:SequentialCrewAIAPIBankAdapter",
        "centralized": "optimizers.bridge.adapters.module_apibank:CentralizedAPIBankAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_apibank:CentralizedAutoGenAPIBankAdapter",
    },
    "toolhop": {
        "single": "optimizers.bridge.adapters.module_toolhop:SingleToolHopAdapter",
        "independent": "optimizers.bridge.adapters.module_toolhop:IndependentToolHopAdapter",
        "decentralized": "optimizers.bridge.adapters.module_toolhop:DecentralizedToolHopAdapter",
        "decentralized_openai_agents": "optimizers.bridge.adapters.module_toolhop:DecentralizedOpenAIAgentsToolHopAdapter",
        "sequential": "optimizers.bridge.adapters.module_toolhop:SequentialToolHopAdapter",
        "sequential_crewai": "optimizers.bridge.adapters.module_toolhop:SequentialCrewAIToolHopAdapter",
        "centralized": "optimizers.bridge.adapters.module_toolhop:CentralizedToolHopAdapter",
        "centralized_autogen": "optimizers.bridge.adapters.module_toolhop:CentralizedAutoGenToolHopAdapter",
    },
}

_TEAMSIZES_DATASETS = ("hotpotqa", "lcb", "bfcl")
_TEAMSIZES_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
TEAM_SIZES = (2, 4, 8, 10)  # canonical team-size sweep — single source of truth
_COMMUNICATIONS_DATASETS = ("hotpotqa", "lcb", "bfcl")
_COMMUNICATIONS_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
_COMMUNICATIONS_FORMATS = ("freeform", "semi_structured", "structured_soft")


def _teamsizes_class_name(dataset: str, base_topology: str, team_size: int) -> str:
    dataset_prefix = {"hotpotqa": "HotpotQATeamSizes", "lcb": "LCBTeamSizes", "bfcl": "BFCLTeamSizes"}[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    return f"{dataset_prefix}{topo_prefix}R{team_size}Adapter"


def _communications_class_name(dataset: str, base_topology: str, fmt: str) -> str:
    dataset_prefix = {
        "hotpotqa": "HotpotQACommunications",
        "lcb": "LCBCommunications",
        "bfcl": "BFCLCommunications",
    }[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    fmt_prefix = "".join(part.title() for part in fmt.split("_"))
    return f"{dataset_prefix}{topo_prefix}{fmt_prefix}Adapter"


for _dataset in _TEAMSIZES_DATASETS:
    for _base_topology in _TEAMSIZES_BASE_TOPOLOGIES:
        for _team_size in TEAM_SIZES:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_r{_team_size}"] = (
                "optimizers.bridge.adapters.module_teamsizes:"
                f"{_teamsizes_class_name(_dataset, _base_topology, _team_size)}"
            )

for _dataset in _COMMUNICATIONS_DATASETS:
    for _base_topology in _COMMUNICATIONS_BASE_TOPOLOGIES:
        for _fmt in _COMMUNICATIONS_FORMATS:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_communications_{_fmt}"] = (
                "optimizers.bridge.adapters.module_communications:"
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
                f"optimizers.bridge.adapters.module_{_dataset}:"
                f"{_tooluse_teamsizes_class_name(_dataset, _base_topology, _team_size)}"
            )
        for _fmt in _TOOLUSE_COMMUNICATIONS_FORMATS:
            DATASET_ADAPTERS[_dataset][f"{_base_topology}_communications_{_fmt}"] = (
                f"optimizers.bridge.adapters.module_{_dataset}:"
                f"{_tooluse_communications_class_name(_dataset, _base_topology, _fmt)}"
            )


def datasets() -> list[str]:
    """Return registered dataset names."""
    return sorted(DATASET_ADAPTERS)


def get_adapter_class(dataset: str, topology: str) -> AdapterClass:
    """Return the adapter class for a registered pair."""
    if dataset not in DATASET_ADAPTERS:
        raise KeyError(f"Unknown dataset {dataset!r}; choices={datasets()}")
    adapters = DATASET_ADAPTERS[dataset]
    if topology not in adapters:
        raise KeyError(f"Unknown topology {topology!r} for dataset {dataset!r}; choices={sorted(adapters)}")
    module_name, class_name = adapters[topology].split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, class_name)
