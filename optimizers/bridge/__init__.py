"""Real-runner bridge shared by every prompt optimizer.

``registry`` maps a (dataset, topology) pair to its prompt-mutable adapter
(``adapters/``, interface in ``adapter_protocol``); ``datasets/`` holds the
loaders, fixed splits and metrics; ``lm`` the endpoints and decoding;
``output_contracts`` the protected final-output contracts; ``programs`` and
``mipro_programs`` the DSPy programs as GEPA and MIPRO see them; ``env`` the
environment variables. Importing this package loads none of them. See
``README.md``.
"""
