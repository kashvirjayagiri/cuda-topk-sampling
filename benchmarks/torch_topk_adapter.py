"""Contract-complete PyTorch Top-k adapter; all work is GPU work when inputs are CUDA.
Boundary repair is included in selection timing, never CPU preprocessing.
"""

class CanonicalTopK:
    """Select by descending score, then ascending original token ID.

    PyTorch topk alone does not promise which tied IDs it returns. This adapter
    repairs ties at the kth boundary and orders the selected candidates. All of
    that work belongs to selection time, rather than untimed preparation.
    Buffers are preallocated; PyTorch may use its own internal workspaces.
    Call under torch.no_grad() on the stream where the buffers were created.
    """

    def __init__(self, logits, k):
        import torch
        if logits.ndim != 2 or logits.dtype != torch.float32:
            raise ValueError("logits must be FP32 [B,V]")
        if not logits.is_contiguous() or logits.requires_grad:
            raise ValueError("logits must be contiguous with requires_grad=False")
        self.B, self.V = logits.shape
        if self.B < 1 or self.V > 2147483647 or not 1 <= k <= self.V:
            raise ValueError("Require B>=1 and 1<=k<=V<=INT32_MAX")
        self.logits, self.k, self.torch = logits, k, torch
        options = dict(device=logits.device)
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, **options)
        shape = (self.B, k)
        self.raw_scores = empty(shape, torch.float32)
        self.raw_ids = empty(shape, torch.int64)
        self.threshold = self.raw_scores[:, -1:]
        self.vocab_ids = torch.arange(self.V, dtype=torch.int64, **options)[None, :]
        self.ranks = torch.arange(k, dtype=torch.int64, **options)[None, :]
        self.sentinel = torch.tensor(self.V, dtype=torch.int64, **options)
        self.boundary_mask = empty((self.B, self.V), torch.bool)
        self.boundary_candidates = empty((self.B, self.V), torch.int64)
        self.boundary_ids = empty(shape, torch.int64)
        self.boundary_positions = empty(shape, torch.int64)
        self.strict_mask = empty(shape, torch.bool)
        self.strict_prefix = empty(shape, torch.int64)
        self.strict_count = self.strict_prefix[:, -1:]
        self.slots = empty(shape, torch.int64)
        self.replacements = empty(shape, torch.int64)
        self.selected_boundary = empty(shape, torch.bool)
        self.chosen_ids = empty(shape, torch.int64)
        self.ids_ascending = empty(shape, torch.int64)
        self.id_order = empty(shape, torch.int64)
        self.scores_by_id = empty(shape, torch.float32)
        self.score_order = empty(shape, torch.int64)
        self.ids64 = empty(shape, torch.int64)
        self.scores = empty(shape, torch.float32)
        self.ids = empty(shape, torch.int32)

    def raw_topk(self):
        """Diagnostic only: bare torch.topk, without the sampler's ID contract."""
        self.torch.topk(self.logits, self.k, dim=1, largest=True, sorted=True,
                       out=(self.raw_scores, self.raw_ids))

    def select(self):
        t = self.torch
        self.raw_topk()

        # Find the smallest original IDs among ALL scores equal to the kth
        # score. Ordering only raw_topk's k IDs would miss boundary ties.
        t.eq(self.logits, self.threshold, out=self.boundary_mask)
        t.where(self.boundary_mask, self.vocab_ids, self.sentinel,
                out=self.boundary_candidates)
        t.topk(self.boundary_candidates, self.k, dim=1, largest=False,
               sorted=True, out=(self.boundary_ids, self.boundary_positions))
        t.gt(self.raw_scores, self.threshold, out=self.strict_mask)
        t.cumsum(self.strict_mask, dim=1, dtype=t.int64, out=self.strict_prefix)
        t.sub(self.ranks, self.strict_count, out=self.slots)
        self.slots.clamp_(min=0, max=self.k - 1)
        t.gather(self.boundary_ids, 1, self.slots, out=self.replacements)
        t.eq(self.raw_scores, self.threshold, out=self.selected_boundary)
        t.where(self.selected_boundary, self.replacements, self.raw_ids,
                out=self.chosen_ids)

        # Start in ascending ID order. A STABLE descending score sort then
        # preserves ascending IDs for every group of equal selected scores.
        t.sort(self.chosen_ids, dim=1, out=(self.ids_ascending, self.id_order))
        t.gather(self.logits, 1, self.ids_ascending, out=self.scores_by_id)
        t.sort(self.scores_by_id, dim=1, descending=True, stable=True,
               out=(self.scores, self.score_order))
        t.gather(self.ids_ascending, 1, self.score_order, out=self.ids64)
        self.ids.copy_(self.ids64)

