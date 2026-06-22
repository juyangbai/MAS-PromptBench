"""Real-runner adapter implementations."""

from importlib import import_module


_ADAPTER_IMPORTS = {
    "CentralizedAutoGenBFCLAdapter": "real_runner_gepa.adapters.centralized_autogen_bfcl",
    "CentralizedBFCLAdapter": "real_runner_gepa.adapters.centralized_bfcl",
    "DecentralizedBFCLAdapter": "real_runner_gepa.adapters.decentralized_bfcl",
    "DecentralizedOpenAIBFCLAdapter": "real_runner_gepa.adapters.decentralized_openai_bfcl",
    "IndependentBFCLAdapter": "real_runner_gepa.adapters.independent_bfcl",
    "SequentialBFCLAdapter": "real_runner_gepa.adapters.sequential_bfcl",
    "SequentialCrewAIBFCLAdapter": "real_runner_gepa.adapters.sequential_crewai_bfcl",
    "SingleBFCLAdapter": "real_runner_gepa.adapters.single_bfcl",
}

_BFCL_NAMES = {
    "single": "SingleBFCLAdapter",
    "independent": "IndependentBFCLAdapter",
    "decentralized": "DecentralizedBFCLAdapter",
    "decentralized_openai": "DecentralizedOpenAIBFCLAdapter",
    "sequential": "SequentialBFCLAdapter",
    "sequential_crewai": "SequentialCrewAIBFCLAdapter",
    "centralized": "CentralizedBFCLAdapter",
    "centralized_autogen": "CentralizedAutoGenBFCLAdapter",
}


def _load_adapter(name: str):
    module = import_module(_ADAPTER_IMPORTS[name])
    return getattr(module, name)


def __getattr__(name: str):
    if name == "BFCL_ADAPTERS":
        return {topology: _load_adapter(adapter_name) for topology, adapter_name in _BFCL_NAMES.items()}
    if name in _ADAPTER_IMPORTS:
        return _load_adapter(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BFCL_ADAPTERS",
    "CentralizedAutoGenBFCLAdapter",
    "CentralizedBFCLAdapter",
    "DecentralizedBFCLAdapter",
    "DecentralizedOpenAIBFCLAdapter",
    "IndependentBFCLAdapter",
    "SequentialBFCLAdapter",
    "SequentialCrewAIBFCLAdapter",
    "SingleBFCLAdapter",
]
