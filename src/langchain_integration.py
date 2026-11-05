"""Production-shape LangChain integration for HSTC.

Shows HSTC as a drop-in Runnable that composes with any LangChain chain
via the standard `|` operator. No LLM dependency on the compressor.

Design:
    HybridCompressor.as_runnable()  →  Runnable[str, str]

Usage (actual LangChain code shape):

    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_groq import ChatGroq

    compressor = HybridCompressor.as_runnable()
    prompt = PromptTemplate.from_template("Summarize: {text}")
    llm = ChatGroq(model="openai/gpt-oss-120b")

    chain = (
        {"text": compressor}
        | prompt
        | llm
    )
    chain.invoke("Long article text here...")

This file works WITHOUT installing LangChain — it defines a minimal
Runnable-compatible interface so the compressor can be dropped into any
LangChain chain without a hard LangChain import at compressor load time.

If langchain-core IS installed, the wrapper auto-detects and uses the real
RunnableLambda. If not, it falls back to a duck-typed shim.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from compressor_hstc import hstc_compress
from compressor_hstc_adaptive import hstc_adaptive_compress, detect_workload


class HybridCompressor:
    """LangChain-compatible wrapper for HSTC.

    Behaves like `RunnableLambda` from `langchain_core.runnables` but requires
    no LangChain dependency at import time.
    """

    def __init__(self, adaptive: bool = True, workload: Optional[str] = None):
        """
        Args:
            adaptive: If True, detect workload (news/dialogue/code/rag) and
                      apply tuned stopword set per workload (recommended).
                      If False, use the default news-tuned list only.
            workload: Override auto-detection with a specific workload string.
                      Values: "news", "dialogue", "code", "rag_context".
        """
        self.adaptive = adaptive
        self.workload = workload

    def __call__(self, text: str) -> str:
        """Compress a string. This is the primary interface."""
        if not isinstance(text, str):
            # If LangChain passes a dict like {"text": "..."}, extract
            if isinstance(text, dict) and "text" in text:
                text = text["text"]
            else:
                text = str(text)

        if self.adaptive:
            return hstc_adaptive_compress(text, workload=self.workload)
        else:
            return hstc_compress(text)

    def invoke(self, input: Any, config: Optional[dict] = None) -> str:
        """LangChain Runnable compatibility."""
        return self(input)

    async def ainvoke(self, input: Any, config: Optional[dict] = None) -> str:
        """LangChain async Runnable compatibility."""
        return self(input)

    def batch(self, inputs: list, config: Optional[dict] = None) -> list:
        """LangChain batch processing."""
        return [self(x) for x in inputs]

    def as_runnable(self):
        """Return a langchain_core.runnables.RunnableLambda if LangChain is installed,
        otherwise self (which quacks like one).
        """
        try:
            from langchain_core.runnables import RunnableLambda
            return RunnableLambda(self.__call__)
        except ImportError:
            # Return self — it already has invoke/ainvoke/batch
            return self

    @classmethod
    def for_rag(cls) -> "HybridCompressor":
        """Factory for RAG workloads (long static context + short dynamic user)."""
        return cls(adaptive=True, workload="rag_context")

    @classmethod
    def for_summarization(cls) -> "HybridCompressor":
        """Factory for news/document summarization."""
        return cls(adaptive=True, workload="news")

    @classmethod
    def for_dialogue(cls) -> "HybridCompressor":
        """Factory for conversational/chat workloads (preserves speaker turns)."""
        return cls(adaptive=True, workload="dialogue")


# Convenience singletons
default_compressor = HybridCompressor(adaptive=True)


def demo():
    """End-to-end usage demonstration."""
    article = (
        "The President of the United States announced a new economic plan yesterday. "
        "She told reporters at the White House the plan will take effect in 2026. "
        "Critics argue the plan is too expensive and will worsen the deficit. "
        "Supporters point to projected job growth of 4 million by 2027. "
        "The announcement comes after months of debate in Congress. "
        "Analysts predict the plan will add $2 trillion to the budget. "
        "Markets fell 1 percent on the news."
    )
    print(f"Input ({len(article)}ch):\n  {article[:120]}...\n")

    # Shape 1: Direct call
    compressed = default_compressor(article)
    reduction = (1 - len(compressed) / len(article)) * 100
    print(f"Direct call → {len(compressed)}ch ({reduction:.1f}% reduction)")
    print(f"  {compressed[:120]}...\n")

    # Shape 2: LangChain chain composition
    comp = HybridCompressor.for_summarization().as_runnable()
    print(f"as_runnable() returned: {type(comp).__name__}")

    # Shape 3: Factory methods
    for factory_name, factory in [
        ("for_rag", HybridCompressor.for_rag),
        ("for_summarization", HybridCompressor.for_summarization),
        ("for_dialogue", HybridCompressor.for_dialogue),
    ]:
        c = factory()
        result = c(article)
        r = (1 - len(result) / len(article)) * 100
        print(f"  {factory_name}: {r:.1f}% reduction")

    # Verify invoke/batch/ainvoke all work
    import asyncio
    assert default_compressor.invoke(article) == default_compressor(article)
    assert len(default_compressor.batch([article, article])) == 2
    assert asyncio.run(default_compressor.ainvoke(article)) == default_compressor(article)
    print("\n✓ All Runnable methods (invoke/ainvoke/batch) verified")


if __name__ == "__main__":
    demo()
