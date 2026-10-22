# Vendored from MASPOB, https://github.com/HZ1008/MASPOB, commit 0be3af42da83b9b5ab187c101ec6d2ce87a5fe04,
# file scripts/gnn_model.py. Z. Hong, Q. Zhang, J. Sun, Z. Shang, M. Kong, X. Wang, Y. Shu and Z. Dai,
# "MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks",
# arXiv:2603.02630, 2026. Upstream terms: "This project is released for research purposes. Please refer
# to the paper for details." The pinned upstream tree contains no LICENSE, COPYING or NOTICE file.
# Everything below the end-of-header line is byte-identical to the upstream file; its SHA-256 is pinned
# in ../regime.py (CORE_SOURCE_SHA256) and verified before the file is executed.
# --- end of vendoring header ---
# GNN Model and UCB functions for prompt optimization

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


# =============================================================================
# EDGE INDEX BUILDER
# =============================================================================

def build_edge_index_from_topology(
    topology: List[Dict],
    bidirectional: bool = True
) -> torch.Tensor:
    """Build edge_index (PyG format) from a workflow topology.

    Args:
        topology: Workflow topology; each element contains:
            - name: operator name
            - dependencies: list of dependency operator names
        bidirectional: Whether to add reverse edges.

    Returns:
        edge_index: [2, num_edges] edge list.
    """
    name_to_idx = {topo["name"]: i for i, topo in enumerate(topology)}

    sources = []
    targets = []

    for i, topo in enumerate(topology):
        # Self-loop
        sources.append(i)
        targets.append(i)

        # Build edges from dependencies
        dependencies = topo.get("dependencies", [])
        for dep_name in dependencies:
            if dep_name in name_to_idx:
                j = name_to_idx[dep_name]
                # j → i (dependency → current node)
                sources.append(j)
                targets.append(i)
                # Reverse edge (optional)
                if bidirectional:
                    sources.append(i)
                    targets.append(j)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index


# =============================================================================
# GNN MODEL
# =============================================================================

class WorkflowGAT(nn.Module):
    """Graph Attention Network (GAT) for predicting workflow scores.

    Architecture:
        1. Node Encoder: prompt embedding → hidden_dim
        2. GAT Layers: multi-head attention message passing
        3. Pooling: mean pooling for graph-level representation
        4. MLP: graph representation → predicted score

    Score scaling (score_min, score_max):
        - For high-score datasets (e.g. GSM8K with scores in 0.85-1.0),
          scaling targets from [score_min, score_max] to [0, 1] amplifies
          differences and prevents gradient vanishing.
        - Use scale_score() / unscale_score() to convert during train/inference.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_operators: int,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        topology: List[Dict] = None,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_sigmoid: bool = True,  # set False for high-score datasets like GSM8K
        score_min: float = 0.0,  # lower bound for score scaling
        score_max: float = 1.0,  # upper bound for score scaling
    ):
        super().__init__()
        self.num_operators = num_operators
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.use_sigmoid = use_sigmoid
        self.score_min = score_min
        self.score_max = score_max

        # Node encoder: progressively compress embedding_dim → hidden_dim
        intermediate_dim = min(256, embedding_dim // 8)

        encoder_linear1 = nn.Linear(embedding_dim, intermediate_dim)
        encoder_linear2 = nn.Linear(intermediate_dim, hidden_dim)

        nn.init.kaiming_normal_(encoder_linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(encoder_linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(encoder_linear1.bias)
        nn.init.zeros_(encoder_linear2.bias)

        self.node_encoder = nn.Sequential(
            encoder_linear1,
            nn.ReLU(),
            nn.Dropout(dropout),
            encoder_linear2,
            nn.ReLU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gat_layers.append(GATv2Conv(
                    hidden_dim, hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                ))
            else:
                self.gat_layers.append(GATv2Conv(
                    hidden_dim * num_heads, hidden_dim,
                    heads=num_heads,
                    concat=(i < num_gnn_layers - 1),
                    dropout=dropout,
                ))

        final_gat_dim = hidden_dim if num_gnn_layers > 1 else hidden_dim * num_heads

        # Graph-level MLP
        mlp_linear1 = nn.Linear(final_gat_dim, hidden_dim)
        mlp_linear2 = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_normal_(mlp_linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(mlp_linear2.weight)
        nn.init.zeros_(mlp_linear1.bias)
        nn.init.zeros_(mlp_linear2.bias)

        # Use Sigmoid only when requested; for high-score datasets (0.95+) skip it to avoid gradient vanishing
        if use_sigmoid:
            self.mlp = nn.Sequential(
                mlp_linear1,
                nn.ReLU(),
                nn.Dropout(dropout),
                mlp_linear2,
                nn.Sigmoid(),
            )
        else:
            # Without Sigmoid, use clamp in forward() to bound the output
            self.mlp = nn.Sequential(
                mlp_linear1,
                nn.ReLU(),
                nn.Dropout(dropout),
                mlp_linear2,
            )

        # Build edge index
        if topology is not None:
            edge_index = build_edge_index_from_topology(topology, bidirectional=bidirectional)
        else:
            edge_index = self._build_default_edge_index(num_operators, bidirectional=bidirectional)
        self.register_buffer('edge_index', edge_index)

    def _build_default_edge_index(self, num_nodes: int, bidirectional: bool = True) -> torch.Tensor:
        """Default linear pipeline edges: 0→1→2→...→n-1"""
        sources, targets = [], []
        for i in range(num_nodes):
            sources.append(i)
            targets.append(i)
            if i < num_nodes - 1:
                sources.append(i)
                targets.append(i + 1)
                if bidirectional:
                    sources.append(i + 1)
                    targets.append(i)
        return torch.tensor([sources, targets], dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        node_features = x.view(batch_size, self.num_operators, self.embedding_dim)

        outputs = []
        for b in range(batch_size):
            h = node_features[b]
            h = self.node_encoder(h)

            for gat_layer in self.gat_layers:
                h = gat_layer(h, self.edge_index)
                h = F.relu(h)

            graph_repr = h.mean(dim=0, keepdim=True)
            score = self.mlp(graph_repr)

            # Without Sigmoid, clamp output to [0, 1]
            if not self.use_sigmoid:
                score = torch.clamp(score, 0.0, 1.0)

            outputs.append(score.squeeze())

        result = torch.stack(outputs)
        return result.squeeze(-1) if result.dim() > 1 else result

    def scale_score(self, score: torch.Tensor) -> torch.Tensor:
        """Scale raw score from [score_min, score_max] to [0, 1].

        Used during training to map targets to a range easier for the GNN to learn.
        E.g., for GSM8K (0.85-1.0), 0.92 → 0.47.
        """
        if self.score_min == 0.0 and self.score_max == 1.0:
            return score  # no scaling needed
        score_range = self.score_max - self.score_min
        if score_range < 1e-6:
            return score  # avoid division by zero
        return (score - self.score_min) / score_range

    def unscale_score(self, scaled_score: torch.Tensor) -> torch.Tensor:
        """Inverse-scale from [0, 1] back to [score_min, score_max].

        Used during inference to convert GNN output to the original score range.
        """
        if self.score_min == 0.0 and self.score_max == 1.0:
            return scaled_score  # no scaling needed
        score_range = self.score_max - self.score_min
        return scaled_score * score_range + self.score_min

    def reset_parameters(self):
        """Reset all model parameters to their initial state."""
        # Reset node encoder
        for module in self.node_encoder:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)

        # Reset GAT layers
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()

        # Reset MLP
        for i, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                if i == len(self.mlp) - 2:  # second-to-last Linear layer
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_gradient_feature(
    frozen_model: WorkflowGAT,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Compute the gradient feature vector using frozen initial parameters."""
    frozen_model.eval()
    embedding = embedding.clone().detach().requires_grad_(False)
    frozen_model.zero_grad()

    for param in frozen_model.parameters():
        param.requires_grad_(True)

    pred = frozen_model(embedding)
    pred.backward()

    grad_parts = []
    for param in frozen_model.parameters():
        if param.grad is not None:
            grad_parts.append(param.grad.view(-1).clone())

    frozen_model.zero_grad()

    if grad_parts:
        return torch.cat(grad_parts)
    else:
        return torch.zeros(1, device=embedding.device)


# =============================================================================
# UCB FUNCTIONS
# =============================================================================

def initialize_fisher(
    ucb_type: str,
    frozen_model: WorkflowGAT = None,
    embedding_dim: int = None,
    device: torch.device = None,
    lambda_reg: float = 1.0,
) -> torch.Tensor:
    """Initialize the Fisher information matrix for the given UCB type."""
    if ucb_type == "neural":
        if frozen_model is None:
            raise ValueError("frozen_model is required for neural UCB")
        num_params = sum(p.numel() for p in frozen_model.parameters())
        dev = next(frozen_model.parameters()).device
        return torch.ones(num_params, device=dev) * lambda_reg
    elif ucb_type == "linear":
        if embedding_dim is None or device is None:
            raise ValueError("embedding_dim and device are required for linear UCB")
        return torch.eye(embedding_dim, device=device) * lambda_reg
    else:  # greedy
        return None


def update_fisher(
    ucb_type: str,
    fisher_matrix: torch.Tensor,
    feature_vector: torch.Tensor,
    fisher_coef: float = 10,
) -> torch.Tensor:
    """Update the Fisher information matrix.

    Args:
        ucb_type: UCB type ('neural', 'linear', 'greedy').
        fisher_matrix: Current Fisher information matrix.
        feature_vector: Feature vector.
        fisher_coef: Update coefficient A = A + coef * x * x^T.
                     coef > 1: faster uncertainty decrease (more exploitation).
                     coef < 1: slower uncertainty decrease (more exploration).
    """
    if ucb_type == "neural":
        return fisher_matrix + fisher_coef * (feature_vector ** 2)
    elif ucb_type == "linear":
        return fisher_matrix + fisher_coef * torch.outer(feature_vector, feature_vector)
    else:  # greedy
        return fisher_matrix


def compute_prediction_and_uncertainty(
    ucb_type: str,
    trained_model: WorkflowGAT,
    embedding: torch.Tensor,
    frozen_model: WorkflowGAT = None,
    fisher_matrix: torch.Tensor = None,
) -> tuple:
    """Compute predicted score and uncertainty for the given UCB type.

    The returned prediction is inverse-scaled to [score_min, score_max].
    """
    trained_model.eval()
    with torch.no_grad():
        scaled_pred = trained_model(embedding)
        # Inverse-scale from [0,1] to [score_min, score_max]
        pred_score = trained_model.unscale_score(scaled_pred).item()

    if ucb_type == "greedy":
        return pred_score, 0.0, None

    elif ucb_type == "neural":
        grad_vector = compute_gradient_feature(frozen_model, embedding)
        uncertainty = torch.sqrt(torch.sum(grad_vector ** 2 / fisher_matrix)).item()
        return pred_score, uncertainty, grad_vector

    elif ucb_type == "linear":
        A_inv_x = torch.linalg.solve(fisher_matrix, embedding)
        uncertainty = torch.sqrt(torch.dot(embedding, A_inv_x)).item()
        return pred_score, uncertainty, embedding

    else:
        raise ValueError(f"Unknown ucb_type: {ucb_type}")


def build_combined_embedding(
    all_operator_embeddings: List[torch.Tensor],
    prompt_indices: List[int],
) -> torch.Tensor:
    """Build a combined embedding by concatenating selected prompt embeddings."""
    return torch.cat([
        all_operator_embeddings[i][prompt_indices[i]]
        for i in range(len(all_operator_embeddings))
    ], dim=0)


def select_best_prompt_for_operator(
    ucb_type: str,
    trained_model: WorkflowGAT,
    operator_idx: int,
    current_prompt_indices: List[int],
    all_operator_embeddings: List[torch.Tensor],
    frozen_model: WorkflowGAT = None,
    fisher_matrix: torch.Tensor = None,
    alpha: float = 0.1,
) -> tuple:
    """Select the best prompt for a given operator using UCB (vectorized).

    Returns inverse-scaled predicted scores in the original score range.
    """
    num_prompts = all_operator_embeddings[operator_idx].shape[0]
    num_operators = len(all_operator_embeddings)

    # Build all candidate embeddings in batch
    batch_parts = []
    for op_idx in range(num_operators):
        if op_idx == operator_idx:
            batch_parts.append(all_operator_embeddings[operator_idx])
        else:
            fixed_emb = all_operator_embeddings[op_idx][current_prompt_indices[op_idx]]
            batch_parts.append(fixed_emb.unsqueeze(0).expand(num_prompts, -1))

    batch_combined = torch.cat(batch_parts, dim=1)

    # Batch-compute predicted scores (model outputs scaled scores)
    trained_model.eval()
    with torch.no_grad():
        scaled_pred_scores = trained_model(batch_combined).squeeze()
        # Inverse-scale back to the original score range
        pred_scores = trained_model.unscale_score(scaled_pred_scores)

    if pred_scores.dim() == 0:
        pred_scores = pred_scores.unsqueeze(0)

    # Compute uncertainties
    if ucb_type == "greedy":
        uncertainties = torch.zeros(num_prompts, device=pred_scores.device)
    elif ucb_type == "linear":
        A_inv_batch = torch.linalg.solve(fisher_matrix, batch_combined.T).T
        uncertainties = torch.sqrt(torch.sum(batch_combined * A_inv_batch, dim=1))
    elif ucb_type == "neural":
        uncertainties = torch.zeros(num_prompts, device=pred_scores.device)
        for i in range(num_prompts):
            grad_vector = compute_gradient_feature(frozen_model, batch_combined[i])
            uncertainties[i] = torch.sqrt(torch.sum(grad_vector ** 2 / fisher_matrix))
    else:
        raise ValueError(f"Unknown ucb_type: {ucb_type}")

    # Compute UCB scores and find the best (using inverse-scaled scores)
    ucb_scores = pred_scores + alpha * uncertainties
    best_idx = ucb_scores.argmax().item()

    return (
        best_idx,
        pred_scores[best_idx].item(),
        uncertainties[best_idx].item(),
        ucb_scores[best_idx].item()
    )


def _extract_placeholders_gnn(template: str) -> set:
    """Extract all {placeholder} patterns from a template."""
    import re
    return set(re.findall(r'\{(\w+)\}', template))


def inject_prompt_to_workflow(flow, topology: List[Dict], prompt_indices: List[int], prompts: List[List[str]]):
    """Inject selected prompts into workflow based on topology.

    Also validates that required placeholders are present in the candidate prompt.
    If validation fails, falls back to the original prompt (index 0).
    """
    for op_idx, topo_item in enumerate(topology):
        attr_path = topo_item["attr_path"]
        candidate_idx = prompt_indices[op_idx]
        prompt_text = prompts[op_idx][candidate_idx]

        # Get the original prompt (index 0) for placeholder comparison
        original_prompt = prompts[op_idx][0]
        original_placeholders = _extract_placeholders_gnn(original_prompt)
        candidate_placeholders = _extract_placeholders_gnn(prompt_text)

        # Check if all required placeholders are present
        missing = original_placeholders - candidate_placeholders
        # Check if candidate has EXTRA placeholders that original doesn't have
        extra = candidate_placeholders - original_placeholders

        if missing or extra:
            if missing:
                print(f"  [Warning] {topo_item['name']} candidate {candidate_idx} missing placeholders: {missing}, using original")
            if extra:
                print(f"  [Warning] {topo_item['name']} candidate {candidate_idx} has extra placeholders: {extra}, using original")
            prompt_text = original_prompt  # Fallback to original

        parts = attr_path.split(".")
        obj = flow
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], prompt_text)

