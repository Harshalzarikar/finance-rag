from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

from src.core.faithfulness_guard import FaithfulnessGuard, FaithfulnessResult


class MockFaithfulLLM(FakeListLLM):
    def with_structured_output(self, *args, **kwargs):
        return RunnableLambda(lambda x: FaithfulnessResult(is_faithful=True))


class MockUnfaithfulLLM(FakeListLLM):
    def with_structured_output(self, *args, **kwargs):
        return RunnableLambda(lambda x: FaithfulnessResult(is_faithful=False))


def test_faithfulness_guard_true():
    fake_llm = MockFaithfulLLM(responses=[])
    guard = FaithfulnessGuard(llm=fake_llm)

    is_faithful = guard.check(query="What is the sky?", context="The sky is blue.", answer="The sky is blue.")

    assert is_faithful is True


def test_faithfulness_guard_false():
    fake_llm = MockUnfaithfulLLM(responses=[])
    guard = FaithfulnessGuard(llm=fake_llm)

    is_faithful = guard.check(query="What is the sky?", context="The sky is blue.", answer="The sky is red.")

    assert is_faithful is False


def test_faithfulness_guard_empty_answer():
    fake_llm = FakeListLLM(responses=[])
    guard = FaithfulnessGuard(llm=fake_llm)

    assert guard.check("query", "context", "") is True
    assert guard.check("query", "context", "   ") is True
