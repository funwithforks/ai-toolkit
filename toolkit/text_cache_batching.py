"""Token-budget packing for the batched text-embedding cache (phase 2).

The shared cache pass runs one TE forward per caption; the tower is
launch/latency-bound at bs=1, so batching captions is near-linearly
faster. The VRAM cap on batching is a TOTAL-TOKEN budget (model-declared
marginal bytes per padded token, measured on real captions), so packing
is a token problem, not a count problem: sort ascending and greedily fill
groups, giving short captions a large effective bs and long captions a
small one, with no hardcoded batch sizes.
"""


def plan_token_groups(counts_and_payloads, budget_tokens):
    """[(tokens, payload), ...] -> [[(tokens, payload), ...], ...] groups.

    For PADDED batched encode: the next caption joins the open group only
    while ``its_count * (group_size + 1)`` still fits the budget -- its
    own count is the group's new padded max and encoding pads to the
    group max, bounding every group's padded token total by the budget.
    Captions longer than the budget alone still get a group of one
    (encode truncates them anyway).
    """
    budget_tokens = int(budget_tokens)
    if budget_tokens < 1 or not counts_and_payloads:
        return []
    order = sorted(((max(int(t), 1), p) for t, p in counts_and_payloads), key=lambda x: x[0])
    groups = []
    open_group = []
    for tokens, payload in order:
        if open_group and tokens * (len(open_group) + 1) > budget_tokens:
            groups.append(open_group)
            open_group = []
        open_group.append((tokens, payload))
    if open_group:
        groups.append(open_group)
    return groups


def plan_packed_groups(counts_and_payloads, budget_tokens):
    """Same shape, for PACKED (padding-free) encode: the open group stays
    open while the SUM of real token counts fits the budget -- packing
    contributes no padding, so the accounting is exact."""
    budget_tokens = int(budget_tokens)
    if budget_tokens < 1 or not counts_and_payloads:
        return []
    order = sorted(((max(int(t), 1), p) for t, p in counts_and_payloads), key=lambda x: x[0])
    groups = []
    open_group = []
    used = 0
    for tokens, payload in order:
        if open_group and used + tokens > budget_tokens:
            groups.append(open_group)
            open_group = []
            used = 0
        open_group.append((tokens, payload))
        used += tokens
    if open_group:
        groups.append(open_group)
    return groups
