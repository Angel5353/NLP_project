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
    if not chunks:
        return "[No evidence retrieved]"

    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(
            f"[Evidence {i}] (chunk_id={ch['chunk_id']}, doc_id={ch['doc_id']})\n{ch['text']}"
        )
    return "\n\n".join(parts)


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for ch in chunks:
        chunk_id = ch["chunk_id"]
        if chunk_id not in seen:
            deduped.append(ch)
            seen.add(chunk_id)
    return deduped


def parse_sufficiency_label(text: str) -> str:
    normalized = text.strip().lower()

    if normalized == "sufficient":
        return "sufficient"
    if normalized == "insufficient":
        return "insufficient"

    if "insufficient" in normalized:
        return "insufficient"
    if "sufficient" in normalized:
        return "sufficient"

    return "insufficient"


# =========================
# Abstract LLM interface
# =========================

class BaseLLMClient:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def generate_batch(self, prompts: List[str], temperature: float = 0.0) -> List[str]:
        return [self.generate(prompt, temperature=temperature) for prompt in prompts]


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
    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def build_prompt(self, question_item: Dict[str, Any]) -> str:
        return question_item["question"]

    def make_result(self, question_item: Dict[str, Any], answer: str) -> Dict[str, Any]:
        return LLMOnlyResult(
            question_id=question_item["question_id"],
            system="llm_only",
            original_question=question_item["question"],
            final_answer=answer,
        ).to_dict()

    def run(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        answer = self.llm.generate(self.build_prompt(question_item), temperature=0.0)
        return self.make_result(question_item, answer)


class StandardRAGPipeline:
    def __init__(self, retriever: Any, llm: BaseLLMClient, top_k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        return self.retriever.search(question, top_k=self.top_k)

    def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        evidence_text = format_evidence(retrieved_chunks)
        return STANDARD_RAG_PROMPT.format(question=question, evidence=evidence_text)

    def make_result(
        self,
        question_item: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        final_answer: str,
    ) -> Dict[str, Any]:
        return StandardRAGResult(
            question_id=question_item["question_id"],
            system="standard_rag",
            original_question=question_item["question"],
            retrieved_chunks=retrieved_chunks,
            final_answer=final_answer,
        ).to_dict()

    def run(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        retrieved_chunks = self.retrieve(question_item["question"])
        prompt = self.build_prompt(question_item["question"], retrieved_chunks)
        final_answer = self.llm.generate(prompt, temperature=0.0)
        return self.make_result(question_item, retrieved_chunks, final_answer)


class AgenticRAGPipeline:
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

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.retriever.search(query, top_k=top_k or self.top_k)

    def build_sufficiency_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        return SUFFICIENCY_PROMPT.format(
            question=question,
            evidence=format_evidence(retrieved_chunks),
        )

    def build_rewrite_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        return REWRITE_PROMPT.format(
            question=question,
            evidence=format_evidence(retrieved_chunks),
        )

    def prepare_final_evidence(
        self,
        first_round: List[Dict[str, Any]],
        second_round: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged = deduplicate_chunks(first_round + second_round)
        return merged[: self.max_final_evidence]

    def build_final_prompt(self, question: str, final_evidence: List[Dict[str, Any]]) -> str:
        return FINAL_AGENTIC_PROMPT.format(
            question=question,
            evidence=format_evidence(final_evidence),
        )

    def make_result(
        self,
        question_item: Dict[str, Any],
        first_round: List[Dict[str, Any]],
        sufficiency_judgment: str,
        rewritten_query: Optional[str],
        second_round: List[Dict[str, Any]],
        final_evidence: List[Dict[str, Any]],
        final_answer: str,
    ) -> Dict[str, Any]:
        return AgenticRAGResult(
            question_id=question_item["question_id"],
            system="agentic_rag",
            original_question=question_item["question"],
            first_round_retrieval=first_round,
            sufficiency_judgment=sufficiency_judgment,
            rewritten_query=rewritten_query,
            second_round_retrieval=second_round,
            final_evidence=final_evidence,
            used_second_round=len(second_round) > 0,
            final_answer=final_answer,
        ).to_dict()

    def run(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        question = question_item["question"]
        first_round = self.retrieve(question)
        sufficiency_prompt = self.build_sufficiency_prompt(question, first_round)
        sufficiency_raw = self.llm.generate(sufficiency_prompt, temperature=0.0)
        sufficiency_judgment = parse_sufficiency_label(sufficiency_raw)

        rewritten_query: Optional[str] = None
        second_round: List[Dict[str, Any]] = []
        if sufficiency_judgment == "insufficient":
            rewrite_prompt = self.build_rewrite_prompt(question, first_round)
            rewritten_query = self.llm.generate(rewrite_prompt, temperature=0.0).strip()
            second_round = self.retrieve(rewritten_query)

        final_evidence = self.prepare_final_evidence(first_round, second_round)
        final_prompt = self.build_final_prompt(question, final_evidence)
        final_answer = self.llm.generate(final_prompt, temperature=0.0)

        return self.make_result(
            question_item=question_item,
            first_round=first_round,
            sufficiency_judgment=sufficiency_judgment,
            rewritten_query=rewritten_query,
            second_round=second_round,
            final_evidence=final_evidence,
            final_answer=final_answer,
        )
