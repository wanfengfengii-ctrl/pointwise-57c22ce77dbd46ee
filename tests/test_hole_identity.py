import unittest

from delivery.analysis_store import ArtifactStore
from delivery.database import ArchiveDatabase
from network.revisions import RevisionRepository
from quality.reports import ReportService
from study.cohorts import StudyService
from topology.inclusions import AnalysisService


class HoleIdentityIncident(unittest.TestCase):
    def analyze(self, versions):
        with ArchiveDatabase.open(':memory:') as db:
            revisions = RevisionRepository(db)
            refs = [revisions.save_network('ridge', index, list(range(1, 8)), edges, faces)
                    for index, (edges, faces) in enumerate(versions, 1)]
            studies = StudyService(db)
            study = studies.freeze_study('morning', [(10 * index, ref) for index, ref in enumerate(refs, 1)])
            store = ArtifactStore(db)
            analysis = AnalysisService(studies, store).analyze_study(study.study_id, 'review')
            fixed = store.read_artifact(analysis.artifact_id)
            report = ReportService(store).create_report(fixed.artifact_id, 'crew')
        return fixed.payload, report.payload

    def test_fixed_analysis_and_report(self):
        a = [(1, 3), (1, 4), (3, 4)]
        b = [(5, 6), (5, 7), (6, 7)]
        versions = [(a, []), (a + b + [(2, 3), (2, 4)], [(2, 3, 4)]), (b, [])]
        analysis, report = self.analyze(versions)
        dimensions = [row['holes'] for row in analysis['summaries']]
        incoming = [row['columns'][0] for row in analysis['maps'] if row['target'] == 2]
        self.assertEqual(dimensions, [1, 1, 2, 1, 1])
        with self.subTest(result='frozen relations'):
            self.assertTrue(all(incoming) and len(set(incoming)) == 2,
                            f'HOLE_IDENTITY_DRIFT: middle position must receive two distinct nonzero lines, got {incoming}')
        with self.subTest(result='published lifetimes'):
            self.assertEqual(report['intervals'], [[0, 2, 1], [2, 4, 1]], 'HOLE_IDENTITY_DRIFT')

    def test_opposite_faces_close_different_cycles(self):
        edges = [(1, 3), (1, 4), (3, 4), (5, 6), (5, 7), (6, 7)]
        analysis, report = self.analyze([(edges, [(5, 6, 7)]), (edges, [(1, 3, 4)])])
        self.assertEqual([row['holes'] for row in analysis['summaries']], [1, 2, 1])
        self.assertEqual(report['intervals'], [[0, 1, 1], [1, 2, 1]], 'HOLE_IDENTITY_DRIFT')
