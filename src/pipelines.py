from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


# =========================
# Prompt templates
# =========================

STANDARD_RAG_PROMPT = """You are a legal QA assistant.

Answer the question using only the provided evidence.
If the evidence is insufficient, say so clearly.
Do not make unsupported claims.

Question:
{question}

Evidence:
{evidence}
"""

SUFFICIENCY_PROMPT = """You are judging whether the retrieved legal evidence is sufficient to answer the question reliably.

Question:
{question}

Retrieved evidence:
{evidence}

Reply with exactly one word:
sufficient
or
insufficient
"""

REWRITE_PROMPT = """Rewrite the following legal question into a short retrieval-oriented query.

Keep the meaning unchanged.
Focus on legal concepts, clause names, obligations, permissions, entities, or conditions.

Original question:
{question}

Retrieved evidence:
{evidence}

Return only the rewritten query.
"""

FINAL_AGENTIC_PROMPT = """You are a legal QA assistant.

Use only the evidence below to answer the question.
Do not add unsupported claims.
If the evidence is insufficient, say so explicitly.

Question:
{question}

Evidence:
{evidence}
"""


# =========================
# Helper functions
# =========================

def format_evidence(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a prompt-friendly evidence block.
    """
    if not chunks:
        return "[No evidence retrieved]"

    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(
            f"[Evidence {i}] (chunk_id={ch['chunk_id']}, doc_id={ch['doc_id']})\n{ch['text']}"
        )
    return "\n\n".join(parts)


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate by chunk_id, preserving order.
    """
    seen = set()
    deduped = []
    for ch in chunks:
        chunk_id = ch["chunk_id"]
        if chunk_id not in seen:
            deduped.append(ch)
            seen.add(chunk_id)
    return deduped


def parse_sufficiency_label(text: str) -> str:
    """
    Robust parser for the sufficiency judgment.
    """
    normalized = text.strip().lower()

    if normalized == "sufficient":
        return "sufficient"
    if normalized == "insufficient":
        return "insufficient"

    # Fallback handling if the model returns extra text.
    if "insufficient" in normalized:
        return "insufficient"
    if "sufficient" in normalized:
        return "sufficient"

    # Conservative fallback
    return "insufficient"


# =========================
# Abstract LLM interface
# =========================

class BaseLLMClient:
    """
    Minimal interface expected by the pipelines.

    You can later replace this with your OpenAI wrapper.
    """

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


# =========================
# Output dataclasses
# =========================

@dataclass
class LLMOnlyResult:
    question_id: str
    system: str
    original_question: str
    final_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StandardRAGResult:
    question_id: str
    system: str
    original_question: str
    retrieved_chunks: List[Dict[str, Any]]
    final_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgenticRAGResult:
    question_id: str
    system: str
    original_question: str
    first_round_retrieval: List[Dict[str, Any]]
    sufficiency_judgment: str
    rewritten_query: Optional[str]
    second_round_retrieval: List[Dict[str, Any]]
    final_evidence: List[Dict[str, Any]]
    used_second_round: bool
    final_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# Pipelines
# =========================

class LLMOnlyPipeline:
    """
    Baseline with no external retrieval.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def run(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        question = question_item["question"]
        question_id = question_item["question_id"]

        answer = self.llm.generate(question, temperature=0.0)

        result = LLMOnlyResult(
            question_id=question_id,
            system="llm_only",
            original_question=question,
            final_answer=answer,
        )
        return result.to_dict()


class StandardRAGPipeline:
    """
    Conventional retrieve-then-generate pipeline.
    """

    def __init__(
        self,
        retriever: Any,
        llm: BaseLLMClient,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        return self.retriever.search(question, top_k=self.top_k)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        evidence_text = format_evidence(retrieved_chunks)
        prompt = STANDARD_RAG_PROMPT.format(
            question=question,
            evidence=evidence_text,
        )
        return self.llm.generate(prompt, temperature=0.0)

    def run(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        question = question_item["question"]
        question_id = question_item["question_id"]

        retrieved_chunks = self.retrieve(question)
        final_answer = self.generate_answer(question, retrieved_chunks)

        result = StandardRAGResult(
            question_id=question_id,
            system="standard_rag",
            original_question=question,
            retrieved_chunks=retrieved_chunks,
            final_answer=final_answer,
        )
        return result.to_dict()


class AgenticRAGPipeline:
    """
    Simple controlled two-round agentic retrieval pipeline.

    Workflow:
        1. Retrieve top-k with original question
        2. Judge sufficiency
        3. If insufficient:
              rewrite query
              retrieve second top-k
        4. Merge + deduplicate evidence
        5. Generate final answer
    """

    def __init__(
        self,
        retriever: Any,
        llm: BaseLLMClient,
        top_k: int = 5,
        max_final_evidence: int = 8,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_final_evidence = max_final_evidence

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        return self.retriever.search(query, top_k=self.top_k)

    def judge_sufficiency(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        prompt = SUFFICIENCY_PROMPT.format(
            question=question,
            evidence=format_evidence(retrieved_chunks),
        )
        raw_output = self.llm.generate(prompt, temperature=0.0)
        return parse_sufficiency_label(raw_output)

    def rewrite_query(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        prompt = REWRITE_PROMPT.format(
            question=question,
            evidence=format_evidence(retrieved_chunks),
        )
        rewritten = self.llm.generate(prompt, temperature=0.0).strip()
        return rewritten

    def prepare_final_evidence(
        self,
        first_round: List[Dict[str, Any]],
        second_round: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged = deduplicate_chunks(first_round + second_round)
        return merged[: self.max_final_evidence]

    def generate_final_answer(
        self,
        question: str,
        final_evidence: List[Dict[str, Any]],
    ) -> str:
        prompt = FINAL_AGENTIC_PROMPT.format(
            question=question,
            evidence=format_evidence(final_evidence),
        )
        return self.llm.generate(prompt, temperature=0.0)

    def run(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        question = question_item["question"]
        question_id = question_item["question_id"]

        # Round 1
        first_round = self.retrieve(question)

        # Decision
        sufficiency_judgment = self.judge_sufficiency(question, first_round)

        rewritten_query: Optional[str] = None
        second_round: List[Dict[str, Any]] = []

        if sufficiency_judgment == "insufficient":
            rewritten_query = self.rewrite_query(question, first_round)
            second_round = self.retrieve(rewritten_query)

        final_evidence = self.prepare_final_evidence(first_round, second_round)
        final_answer = self.generate_final_answer(question, final_evidence)

        result = AgenticRAGResult(
            question_id=question_id,
            system="agentic_rag",
            original_question=question,
            first_round_retrieval=first_round,
            sufficiency_judgment=sufficiency_judgment,
            rewritten_query=rewritten_query,
            second_round_retrieval=second_round,
            final_evidence=final_evidence,
            used_second_round=len(second_round) > 0,
            final_answer=final_answer,
        )
        return result.to_dict()