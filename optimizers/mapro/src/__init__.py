"""MAPRO core: prompt graph, exact MAP inference, rewards, demonstrations and refinement.

``mas.graph`` holds the prompt graph, ``infer.bp`` exact max-product MAP,
``reward`` the node/edge judges and demonstration pools, ``refine`` blame
feedback and pool mutation; ``llm`` is the chat surface they call.
"""
