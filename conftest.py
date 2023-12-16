import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """This fixture is used to capture logs from the logger.

    Override the default pytest caplog fixture to use the loguru logger.

    Source:
    `Loguru documentation
    <https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library>`_

    """
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        # Set to 'True' if your test is spawning child processes.
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)
