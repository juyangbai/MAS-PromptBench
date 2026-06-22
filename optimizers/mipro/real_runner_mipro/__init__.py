"""Real-runner MIPRO experiment workspace."""

__all__ = [
    "AdapterBackedProgram",
    "BFCLRealRunnerProgram",
    "DATASET_ADAPTERS",
    "IndependentBFCLRealRunnerProgram",
    "MIPROAdapterBackedProgram",
    "MIPRORealRunnerProgram",
    "RealRunnerProgram",
    "adapter_choices",
    "datasets",
    "get_adapter_class",
    "topologies",
]


def __getattr__(name: str):
    """Load heavier DSPy/framework dependencies only when requested."""
    if name in {
        "AdapterBackedProgram",
        "BFCLRealRunnerProgram",
        "IndependentBFCLRealRunnerProgram",
        "RealRunnerProgram",
    }:
        from real_runner_mipro import programs

        return getattr(programs, name)
    if name in {"MIPROAdapterBackedProgram", "MIPRORealRunnerProgram"}:
        from real_runner_mipro import mipro_programs

        return getattr(mipro_programs, name)
    if name in {
        "DATASET_ADAPTERS",
        "adapter_choices",
        "datasets",
        "get_adapter_class",
        "topologies",
    }:
        from real_runner_mipro import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
