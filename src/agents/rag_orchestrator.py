"""
RAG Orchestrator that integrates LLM with retrieval system.
Handles answer generation, tool usage, and multi-step reasoning.
"""
from typing import Dict, Any, Optional, List
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from loguru import logger

from src.agents.calculator_tool import FinancialCalculatorTool
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.prompts import RAG_SYSTEM_PROMPT, PromptTemplates
from src.utils.citation_engine import CitationEngine
from src.config import get_settings


class RAGOrchestrator:
    """
    Orchestrates the RAG pipeline with LLM and tools.

    Components:
    - Retrieval system (hybrid search)
    - LLM for answer generation
    - Calculator tool for computations
    - Citation engine for source tracking
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        citation_engine: CitationEngine,
        llm_model: str = "claude-3-5-sonnet-20241022",
        llm_provider: str = "anthropic",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Initialize the RAG orchestrator.

        Args:
            hybrid_retriever: Hybrid retrieval system
            citation_engine: Citation tracking system
            llm_model: Model name (e.g., "claude-3-5-sonnet-20241022" or "gpt-4o-mini")
            llm_provider: "anthropic" or "openai"
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
        """
        self.settings = get_settings()
        self.hybrid_retriever = hybrid_retriever
        self.citation_engine = citation_engine
        self.logger = logger.bind(module="rag_orchestrator")

        # Initialize LLM based on provider
        if llm_provider.lower() == "anthropic":
            self.llm = ChatAnthropic(
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=self.settings.anthropic_api_key
            )
        else:  # openai
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.settings.openai_api_key
            )

        # Initialize calculator tool
        self.calculator = FinancialCalculatorTool()
        self.tools = [self.calculator.create_langchain_tool()]

        # Create agent
        self.agent_executor = self._create_agent()

        self.logger.info(f"RAG Orchestrator initialized with model: {llm_model}")

    def _create_agent(self):
        """
        Create the LangGraph agent with tools.

        Returns:
            Compiled agent graph
        """
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=RAG_SYSTEM_PROMPT
        )

    def answer_question(
        self,
        question: str,
        top_k_vector: int = 10,
        top_k_bm25: int = 10,
        top_k_final: int = 5,
        include_calculations: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.

        Args:
            question: User's question
            top_k_vector: Number of vectors to retrieve
            top_k_bm25: Number of BM25 results to retrieve
            top_k_final: Final number of chunks to use
            include_calculations: Whether to use calculator tool

        Returns:
            Dictionary with answer, citations, and metadata
        """
        self.logger.info(f"Processing question: {question}")

        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.hybrid_retriever.retrieve(
                query=question,
                top_k_vector=top_k_vector,
                top_k_bm25=top_k_bm25,
                top_k_final=top_k_final
            )

            if not retrieved_chunks:
                return {
                    "answer": "NOT_FOUND: No relevant information found in the document for this query.",
                    "citations": [],
                    "calculation_steps": None,
                    "confidence": 0.0,
                    "status": "no_results"
                }

            # Step 2: Format context for LLM
            context = self._format_context(retrieved_chunks)

            # Step 3: Determine query type and create prompt
            prompt_input = self._create_prompt_input(question, context)

            # Step 4: Generate answer with LLM (and tools if needed)
            if include_calculations and self._requires_calculation(question):
                # Use agent with tools
                result = self.agent_executor.invoke(
                    {"messages": [HumanMessage(content=prompt_input)]}
                )
                answer = result["messages"][-1].content
                calculation_steps = self._extract_calculation_steps(result)
            else:
                # Direct LLM call without tools
                messages = [
                    SystemMessage(content=RAG_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_input)
                ]
                response = self.llm.invoke(messages)
                answer = response.content
                calculation_steps = None

            # Step 5: Generate citations
            citations = self.citation_engine.create_citations(
                retrieved_chunks=retrieved_chunks,
                answer=answer
            )

            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(retrieved_chunks, answer)

            return {
                "answer": answer,
                "citations": citations,
                "calculation_steps": calculation_steps,
                "confidence": confidence,
                "status": "success",
                "chunks_retrieved": len(retrieved_chunks)
            }

        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"ERROR: An error occurred while processing your question: {str(e)}",
                "citations": [],
                "calculation_steps": None,
                "confidence": 0.0,
                "status": "error"
            }

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            section = chunk.get("section", "Unknown")
            page = chunk.get("page_number", "Unknown")
            content = self._clean_chunk_content(chunk.get("content", ""))
            chunk_type = chunk.get("chunk_type", "text")

            context_parts.append(
                f"[Source {i}] (Page {page}, {section}) [{chunk_type}]\n{content}\n"
            )

        return "\n".join(context_parts)

    def _clean_chunk_content(self, content: str) -> str:
        """Remove repeated PDF header noise from chunk content."""
        import re
        # Remove repeated markdown headers like "## Apple Inc. | 2025 Form 10-K | 18"
        content = re.sub(r'(## Apple Inc\. \| \d{4} Form 10-K \| \d+\s*\n?)+', '', content)
        # Collapse multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()

    def _create_prompt_input(self, question: str, context: str) -> str:
        """
        Create prompt input based on question type.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        # Detect query type
        if self._requires_calculation(question):
            return PromptTemplates.financial_calculation_template(question, context)
        elif self._is_multi_step(question):
            return PromptTemplates.multi_step_template(question, context)
        else:
            return PromptTemplates.factual_retrieval_template(question, context)

    def _requires_calculation(self, question: str) -> bool:
        """
        Detect if question requires calculations.

        Args:
            question: User's question

        Returns:
            True if calculation is needed
        """
        calculation_keywords = [
            "calculate", "compute", "what is the percentage",
            "growth rate", "change", "difference", "increase",
            "decrease", "how much", "compare", "ratio"
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in calculation_keywords)

    def _is_multi_step(self, question: str) -> bool:
        """
        Detect if question requires multi-step reasoning.

        Args:
            question: User's question

        Returns:
            True if multi-step reasoning is needed
        """
        multi_step_keywords = [
            "analyze", "explain", "compare", "contrast",
            "evaluate", "assess", "how and why"
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in multi_step_keywords)

    def _extract_calculation_steps(self, agent_result: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract calculation steps from agent execution.

        Args:
            agent_result: Result from agent executor

        Returns:
            List of calculation steps or None
        """
        # Get calculation history from calculator
        history = self.calculator.get_calculation_history()

        if not history:
            return None

        steps = []
        for i, calc in enumerate(history, 1):
            steps.append({
                "step": i,
                "code": calc.get("code", ""),
                "result": calc.get("result", ""),
                "success": calc.get("success", False)
            })

        return steps

    def _calculate_confidence(
        self,
        chunks: List[Dict[str, Any]],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the answer.

        Args:
            chunks: Retrieved chunks
            answer: Generated answer

        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0

        # Base confidence on average relevance score
        avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)

        # Penalize if answer says NOT_FOUND
        if "NOT_FOUND" in answer:
            return 0.0

        # Boost if calculations were performed
        if self.calculator.get_calculation_history():
            avg_score = min(avg_score + 0.1, 1.0)

        return round(avg_score, 2)

    def multi_query_answer(
        self,
        questions: List[str],
        top_k_per_query: int = 3
    ) -> Dict[str, Any]:
        """
        Answer multiple related questions and synthesize.

        Args:
            questions: List of related questions
            top_k_per_query: Number of chunks per query

        Returns:
            Synthesized answer
        """
        self.logger.info(f"Processing {len(questions)} related queries")

        all_chunks = []
        sub_answers = []

        # Answer each sub-question
        for question in questions:
            result = self.answer_question(
                question=question,
                top_k_final=top_k_per_query
            )
            sub_answers.append({
                "question": question,
                "answer": result["answer"]
            })

        # Synthesize final answer
        synthesis_prompt = f"""You answered the following sub-questions:

{chr(10).join([f"Q: {sa['question']}\nA: {sa['answer']}\n" for sa in sub_answers])}

Now synthesize these answers into a comprehensive response to the original broad question."""

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt)
        ]

        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sub_answers": sub_answers,
            "status": "success"
        }

    def clear_calculation_history(self):
        """Clear calculator history."""
        self.calculator.clear_history()


def create_rag_orchestrator(
    hybrid_retriever: HybridRetriever,
    citation_engine: CitationEngine,
    model: str = "gpt-4o-mini"
) -> RAGOrchestrator:
    """
    Factory function to create RAG orchestrator.

    Args:
        hybrid_retriever: Hybrid retriever instance
        citation_engine: Citation engine instance
        model: LLM model name

    Returns:
        RAGOrchestrator instance
    """
    return RAGOrchestrator(
        hybrid_retriever=hybrid_retriever,
        citation_engine=citation_engine,
        llm_model=model
    )


if __name__ == "__main__":
    # Test the orchestrator
    print("RAG Orchestrator module loaded successfully")
    print("Use create_rag_orchestrator() to create an instance")
