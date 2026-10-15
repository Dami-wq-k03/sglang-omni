# SPDX-License-Identifier: Apache-2.0
"""Drama 3 scheduler: batch new prefills only while decode is already busy."""

from __future__ import annotations

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_PREFILL_COALESCE_MIN_RUNNING_REQUESTS,
)
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class Drama3Scheduler(OmniScheduler[Drama3RequestData]):
    """Waits for a prefill batch only after decode is already full."""

    prefill_coalesce_min_running_requests: int = (
        DEFAULT_PREFILL_COALESCE_MIN_RUNNING_REQUESTS
    )

    def prefill_coalesce_waiting_requests(
        self, running_batch: ScheduleBatch | None
    ) -> list[Req] | None:
        if running_batch is None or running_batch.is_empty():
            running_request_count = 0
        else:
            running_request_count = len(running_batch.reqs)
        if running_request_count < self.prefill_coalesce_min_running_requests:
            return None
        else:
            return super().prefill_coalesce_waiting_requests(running_batch)
