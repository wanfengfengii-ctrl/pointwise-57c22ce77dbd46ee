from quality.encoding import Interval
from topology.f2 import kernel_basis, rref


def _arrow_view(arrow):
    # Accept both frozen InducedMap objects and validated payload rows.
    if isinstance(arrow, dict):
        return arrow['source'], arrow['target'], arrow['source_dim'], arrow['target_dim'], arrow['columns']
    return arrow.source, arrow.target, arrow.source_dim, arrow.target_dim, arrow.columns


def _row_basis(rows, width):
    basis, _ = rref(tuple(row for row in rows if row), width)
    return basis


def _intersection(left_rows, right_rows, width):
    # Basis for U ∩ V over F2.  Under the standard dot product
    # U ∩ V = (U⊥ + V⊥)⊥: the orthogonal complement of a row space is its
    # kernel, the two complement spans are joined by stacking their bases,
    # and a final kernel recovers the common subspace.
    left_basis = _row_basis(left_rows, width)
    right_basis = _row_basis(right_rows, width)
    if not left_basis or not right_basis:
        return ()
    left_perp = kernel_basis(left_basis, width)
    right_perp = kernel_basis(right_basis, width)
    joined_perp = _row_basis(tuple(left_perp) + tuple(right_perp), width)
    return _row_basis(kernel_basis(joined_perp, width), width)


def subdiagram_rank(dimensions, maps, first, last):
    # Rank of the zigzag subdiagram on positions first..last.  Pack every
    # node's homology space into its own block (exact dimension, no padding),
    # build the relation span R of the arrows and the compatibility space
    # S = ker D of tuples commuting with every arrow.
    window_maps = tuple(_arrow_view(arrow) for arrow in maps
                        if first <= arrow.source <= last and first <= arrow.target <= last)
    offsets = {}
    cursor = 0
    for position in range(first, last + 1):
        offsets[position] = cursor
        cursor += dimensions[position]
    width = cursor

    if not window_maps:
        # No relations: the window is a direct sum of its position spaces.
        return dimensions[first]

    # Relation rows: each arrow contributes (e_i, f(e_i)); columns are packed
    # target-coordinate masks, one per source basis vector.
    relation_rows = []
    # Compatibility rows: row j enforces target coordinate j equals the
    # corresponding source combination, i.e. f(v_source) + v_target = 0.
    compatibility_rows = []
    for source, target, source_dim, target_dim, columns in window_maps:
        source_shift, target_shift = offsets[source], offsets[target]
        for offset, column in enumerate(columns):
            packed = 1 << (source_shift + offset)
            for target_offset in range(target_dim):
                if column & (1 << target_offset):
                    packed ^= 1 << (target_shift + target_offset)
            relation_rows.append(packed)
        for target_offset in range(target_dim):
            row = 1 << (target_shift + target_offset)
            for source_offset, column in enumerate(columns):
                if column & (1 << target_offset):
                    row |= 1 << (source_shift + source_offset)
            compatibility_rows.append(row)

    relation_basis = _row_basis(relation_rows, width)
    compatible_basis = kernel_basis(compatibility_rows, width)
    head_dim = dimensions[first]
    head_mask = (1 << head_dim) - 1
    # The subdiagram rank counts head-projected compatible tuples.  Two head
    # values represent the same class exactly when a relation shifts one into
    # the other while carrying no partner-block (tail) information: only
    # relations supported entirely on the head block identify head elements.
    # Build the head-only ambient subspace, intersect it with the relation
    # span to obtain those pure-head relations, then quotient the head
    # projection of S by their span.
    head_compatible = _row_basis((vector & head_mask for vector in compatible_basis), width)
    head_only_space = tuple(1 << offset for offset in range(head_dim))
    pure_head_relations = _intersection(relation_basis, head_only_space, width)
    overlap = _intersection(head_compatible, pure_head_relations, width)
    return len(head_compatible) - len(overlap)


def decompose_intervals(dimensions, maps):
    count = len(dimensions)
    ranks = {(start, end): subdiagram_rank(dimensions, maps, start, end)
             for start in range(count) for end in range(start, count)}
    intervals = []
    # A subdiagram rank counts interval summands covering the entire window.
    # Möbius inversion at both ends counts summands with exact endpoints.
    for (start, end), rank in ranks.items():
        multiplicity = (rank - ranks.get((start - 1, end), 0) - ranks.get((start, end + 1), 0)
                        + ranks.get((start - 1, end + 1), 0))
        if multiplicity < 0:
            raise ValueError('negative interval multiplicity')
        if multiplicity:
            intervals.append(Interval(start, end, multiplicity))
    return tuple(intervals)


def reconstruct_dimensions(intervals, position_count):
    changes = [0] * (position_count + 1)
    for interval in intervals:
        changes[interval.start] += interval.multiplicity
        changes[interval.end + 1] -= interval.multiplicity
    dimension = 0
    dimensions = []
    for change in changes[:-1]:
        dimension += change
        dimensions.append(dimension)
    return tuple(dimensions)
