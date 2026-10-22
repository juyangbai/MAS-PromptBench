"""MASPOB (bandit prompt optimization with a GNN surrogate) on the shared run protocol.

``integration.MASPOBOptimizer`` is the registered method. ``native`` holds the
retained native components, ``regime`` the frozen settings and pins, and
``upstream/`` the four pinned core files of https://github.com/HZ1008/MASPOB
(arXiv:2603.02630), which the upstream project releases for research purposes.

Extra dependencies (imported lazily): ``pip install numpy torch torch_geometric
sentence-transformers`` (a CPU torch build is enough) and the
``sentence-transformers/all-MiniLM-L6-v2`` weights in the Hugging Face cache.
"""
