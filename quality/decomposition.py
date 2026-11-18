from quality.encoding import Interval
from topology.f2 import matrix_rank


def subdiagram_rank(dimensions, maps, first, last):
    capacity = min(dimensions[first:last + 1])
    for arrow in maps:
        if first <= arrow.source <= last and first <= arrow.target <= last:
            capacity = min(capacity, matrix_rank(arrow.columns))
    return capacity


def decompose_intervals(dimensions, maps):
    count = len(dimensions)
    ranks = {(start, end): subdiagram_rank(dimensions, maps, start, end)
             for start in range(count) for end in range(start, count)}
    intervals = []
    # A subdiagram rank counts interval summands covering the entire window.
    # Take the difference at both ends to count summands with exact endpoints.
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
