"""The job's crawl report model."""

import pytest
from pydantic import ValidationError

from rag_engine.domain import (
    REPORT_SAMPLES_PER_REASON,
    CrawlJob,
    CrawlReport,
    PageReason,
    PageSample,
    PageStatus,
    ReasonCount,
)

# ---- #
# Tests
# ---- #


class TestCrawlReport_S53:
    """S53 — Error accounting: the crawl report model stored on the job."""

    def test_failed_and_unfinished_totals(self) -> None:
        report = CrawlReport(
            pages={PageStatus.FAILED: 3, PageStatus.QUEUED: 1, PageStatus.IN_PROGRESS: 2}
        )

        assert (report.failed, report.unfinished) == (3, 3)
        assert (CrawlReport().failed, CrawlReport().unfinished) == (0, 0)

    def test_samples_are_capped(self) -> None:
        samples = [PageSample(url=f"https://a.example/{n}") for n in range(6)]

        with pytest.raises(ValidationError):
            ReasonCount(status=PageStatus.FAILED, reason=PageReason.TIMEOUT, count=6,
                        samples=samples)  # fmt: skip
        assert REPORT_SAMPLES_PER_REASON == 5

    def test_jobs_have_no_report_until_a_run_writes_one(self) -> None:
        job = CrawlJob(
            tenant_id="ten_" + "1" * 32,
            kb_id="kb_" + "1" * 32,
            source_id="src_" + "1" * 32,
            seed_url="https://a.example/",
        )
        report = CrawlReport(
            failures=[ReasonCount(status=PageStatus.FAILED, reason=PageReason.TIMEOUT, count=1)]
        )

        restored = CrawlJob.from_mongo({**job.to_mongo(), "report": report.model_dump()})

        assert job.report is None
        assert restored.report == report
