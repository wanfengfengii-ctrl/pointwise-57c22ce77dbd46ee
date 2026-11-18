import unittest

from delivery.analysis_store import ArtifactStore
from delivery.database import ArchiveDatabase
from network.revisions import RevisionRepository
from quality.decomposition import subdiagram_rank
from quality.reports import ReportService
from study.cohorts import StudyService
from topology.inclusions import AnalysisService, validate_analysis_relations


class HoleLifetimeRegression(unittest.TestCase):
    def analyze(self, versions, vertices, network='ridge'):
        with ArchiveDatabase.open(':memory:') as db:
            revisions = RevisionRepository(db)
            refs = [revisions.save_network(network, index, vertices, edges, faces)
                    for index, (edges, faces) in enumerate(versions, 1)]
            studies = StudyService(db)
            study = studies.freeze_study('regression', [(10 * index, ref)
                                                         for index, ref in enumerate(refs, 1)])
            store = ArtifactStore(db)
            analysis = AnalysisService(studies, store).analyze_study(study.study_id, 'review')
            report = ReportService(store).create_report(analysis.artifact_id, 'crew')
        return validate_analysis_relations(analysis.payload), report.payload

    def test_transported_cycle_keeps_identity_across_filled_face(self):
        # Hole A stays open in the middle sample even though a solid face is
        # attached along one of its edges; the inclusion must map A to a
        # nonzero homology class rather than an exact representative match.
        a = [(1, 3), (1, 4), (3, 4)]
        b = [(5, 6), (5, 7), (6, 7)]
        middle_edges = a + b + [(2, 3), (2, 4)]
        versions = [(a, []), (middle_edges, [(2, 3, 4)]), (b, [])]
        analysis, report = self.analyze(versions, list(range(1, 8)))
        self.assertEqual(analysis.dimensions, (1, 1, 2, 1, 1))
        incoming = [arrow.columns[0] for arrow in analysis.maps if arrow.target == 2]
        self.assertEqual(len(incoming), 2)
        self.assertTrue(all(incoming) and len(set(incoming)) == 2)
        self.assertEqual(report['intervals'], [[0, 2, 1], [2, 4, 1]])

    def test_distinct_holes_filled_on_opposite_samples(self):
        # Same edge skeleton, different filled faces: each hole closes on a
        # different sample, so their lifetimes meet but do not merge.
        a = [(1, 3), (1, 4), (3, 4)]
        b = [(5, 6), (5, 7), (6, 7)]
        edges = a + b
        versions = [(edges, [(5, 6, 7)]), (edges, [(1, 3, 4)])]
        analysis, report = self.analyze(versions, list(range(1, 8)))
        self.assertEqual(analysis.dimensions, (1, 2, 1))
        self.assertEqual(report['intervals'], [[0, 1, 1], [1, 2, 1]])

    def test_subdiagram_rank_respects_arrow_direction(self):
        # A two-node identity inclusion keeps one surviving class regardless
        # of whether the arrow points into or out of the head node.
        a = [(1, 2), (2, 3), (1, 3)]
        analysis, _ = self.analyze([(a, []), (a, [])], [1, 2, 3])
        maps = analysis.maps
        self.assertEqual(subdiagram_rank(analysis.dimensions, maps, 0, 1), 1)
        self.assertEqual(subdiagram_rank(analysis.dimensions, maps, 1, 2), 1)
        self.assertEqual(subdiagram_rank(analysis.dimensions, maps, 0, 2), 1)

    def test_interval_decomposition_reconstructs_dimensions(self):
        a = [(1, 3), (1, 4), (3, 4)]
        b = [(5, 6), (5, 7), (6, 7)]
        versions = [(a, []), (a + b + [(2, 3), (2, 4)], [(2, 3, 4)]), (b, [])]
        analysis, report = self.analyze(versions, list(range(1, 8)))
        coverage = [0] * len(analysis.dimensions)
        for start, end, multiplicity in report['intervals']:
            for position in range(start, end + 1):
                coverage[position] += multiplicity
        self.assertEqual(coverage, list(analysis.dimensions))


if __name__ == '__main__':
    unittest.main()
