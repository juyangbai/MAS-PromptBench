"""MIPRO (``dspy.teleprompt.MIPROv2``) on the shared run protocol.

``integration.MIPROOptimizer`` is the registered method. It runs the bridge's
``MIPROAdapterBackedProgram`` through the protocol runner with the DSPy
plumbing shared in ``optimizers.protocol.methods`` (``dspy_bridge``,
``context_fit``).
"""
