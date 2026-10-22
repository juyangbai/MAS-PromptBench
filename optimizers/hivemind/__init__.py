"""HiveMind: contribution-guided online prompt optimization (CG-OPO).

Adaptation of "HiveMind: Contribution-Guided Online Prompt Optimization of
LLM Multi-Agent Systems" (arXiv:2512.06432) to the shared run protocol. Each
cycle evaluates worker coalitions of the frozen topology on a training
minibatch (every coalition is a full-MAS rollout with the absent workers
masked), assigns Shapley credit, reflects lessons into the lowest-credit
prompt parameter (the manager every third cycle in centralized teams) and
keeps the update only if it strictly improves a validation minibatch.

* ``optimizer`` — the CG-OPO loop (``HiveMindOptimizer``);
* ``regime`` — settings, coalition plan (exact or Monte-Carlo) and seeds;
* ``topology`` — coalition games, control requests and non-centralized masks;
* ``coalition`` — masked centralized adapters;
* ``runtime`` — the ``optimizer_control`` execution hook and its verification.

The entrypoint is ``integration.HiveMindOptimizer``. Importing it never
contacts a model endpoint; the ``optimizers.bridge`` adapters are imported
only when the first centralized coalition rollout is built.
"""
