from __future__ import annotations

import pytest

from finintel.advisory import StrategicPrescriptiveEngine
from finintel.graph import FinancialKnowledgeGraph
from finintel.ingestion import FixtureSource
from finintel.rag import FinancialRAGIndex

PEERS = ["WMT", "COST", "DG", "KR"]


@pytest.fixture(scope="session")
def bundle():
    return FixtureSource().load()


@pytest.fixture(scope="session")
def index(bundle):
    return FinancialRAGIndex().build_from_bundle(bundle)


@pytest.fixture(scope="session")
def kg(bundle, index):
    return FinancialKnowledgeGraph().build(bundle, index.chunks)


@pytest.fixture(scope="session")
def engine(bundle, index, kg):
    return StrategicPrescriptiveEngine(bundle, index, kg)


@pytest.fixture(scope="session")
def report(engine):
    return engine.run("TGT", PEERS)
