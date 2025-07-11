from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List
import logging
import threading


# Set up logging
logger = logging.getLogger(__name__)


class LLMTimeoutError(Exception):
    """Exception raised when LLM call exceeds timeout limit."""
    pass


# Base for structured outputs
class ResponseFormatter(BaseModel):
    """
    Base model for structured LLM outputs.
    Provides a default response field for all formatters.
    """
    pass

# Specific structured output schemas
class GradeAnswer(ResponseFormatter, BaseModel):
    """
    Response format for grading answers against criteria.
    Used to evaluate the quality of generated responses.
    """
    binary_score: str = Field(default="0", description="Answer addresses the question, '1' or '0'")
    explanation: str = Field(default="", description="Explain the reasoning for the score")

class GradeItem(ResponseFormatter, BaseModel):
    """
    Single evaluation for one supporting document.
    """
    binary_score: Literal['1', '0'] = Field(
        default='0',
        description="Binary score: '1' means relevant; '0' means not relevant."
    )
class GradeCaseReferences(ResponseFormatter, BaseModel):
    """
    Formatter for a list of supporting documents.
    """
    items: List[GradeItem] = Field(
        default=[GradeItem(binary_score='0')],
        description="List of evaluations for each supporting document."
    )
class RouteQuery(ResponseFormatter, BaseModel):
    """
    Response format for routing queries to appropriate data sources.
    Determines whether a question should be answered from documents,
    web search, or general knowledge.
    """
    datasource: Literal["DocumentSearch", "WebSearch", "GENERAL"] = Field(
        default="GENERAL",
        description="Given a user question choose to route it to WebSearch, DocumentSearch, or GENERAL."
    )
    explanation: str = Field(default="", description="Explain the reasoning for the routing decision.")

class CanliiQuery(ResponseFormatter, BaseModel):
    """
    Response format for CanLII database queries.
    Contains the formatted query string for legal database search.
    """
    query: str = Field(default="", description="The formatted query string for CanLII database search")

# Prompt Tool base
class AbstractPromptTool:
    """
    Base class for prompt tools that interact with language models.
    Handles structured output formatting and LLM invocation.
    """
    def __init__(self, llm, llm_instructions: str, response_formatter: type[ResponseFormatter] = None, prompt_template: str = None):
        """
        Initialize the prompt tool with an LLM and response format.
        
        Args:
            llm: Language model to use for generation
            llm_instructions: System instructions for the LLM
            response_formatter: Pydantic model for structured output
            prompt_template: Optional template to format prompts
        """
        self.llm = llm
        self.llm_instructions = llm_instructions
        self.response_formatter = response_formatter
        if response_formatter is not None:
            self.structured_llm_response = llm.with_structured_output(response_formatter)
        else:
            self.structured_llm_response = None
        self.prompt_template = prompt_template

    def _get_default_response(self):
        """
        Get default response when LLM call times out.
        Returns appropriate default based on response formatter type.
        """
        if self.response_formatter is None:
            # For unstructured responses, return a simple timeout message
            return "I'm sorry, I couldn't process your request in time. Please try and rephrase your question or ask a different one."
        
        # For structured responses, create instance with default values
        try:
            return self.response_formatter()
        except Exception:
            # Fallback if default construction fails
            logger.warning(f"Could not create default instance of {self.response_formatter.__name__}")
            return None

    def _invoke_with_timeout(self, messages, timeout=100):
        """
        Invoke LLM with timeout using threading.
        
        Args:
            messages: List of messages to send to LLM
            timeout: Timeout in seconds
            
        Returns:
            LLM response or raises LLMTimeoutError
        """
        result = [None]
        exception = [None]
        
        def target():
            try:
                if self.structured_llm_response is not None:
                    result[0] = self.structured_llm_response.invoke(messages)
                else:
                    result[0] = self.llm.invoke(messages)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Thread is still running, which means it timed out
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise LLMTimeoutError(f"LLM call timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
            
        return result[0]

    def invoke(self, prompt):
        """
        Invoke the LLM with the given prompt and return structured output.
        
        Args:
            prompt: Either a string or dict of values to format with template
            
        Returns:
            Structured output according to the response formatter
        """
        if self.prompt_template is not None:
            # prompt is a dict and must be formatted into a string
            prompt_str = self.prompt_template.format(**prompt)
        else:
            # prompt is already a string
            prompt_str = prompt

        logger.info(f"Invoking LLM with prompt of length: {len(prompt_str)} characters")

        messages = [
            SystemMessage(content=self.llm_instructions),
            HumanMessage(content=prompt_str)
        ]

        
        try:
            result = self._invoke_with_timeout(messages, timeout=100)
            return result
            
        except LLMTimeoutError:
            logger.error("LLM call timed out after 100 seconds, returning default response")
            return self._get_default_response()
        except Exception as e:
            logger.error(f"LLM call failed with error: {str(e)}")
            raise


# Prompt tools with concrete behavior
class RagQuestionRouter(AbstractPromptTool):
    """
    Tool for routing client questions to the appropriate data source:
    DocumentSearch, WebSearch, or GENERAL knowledge response.
    """
    def __init__(self, llm):
        """
        Initialize the question router.

        Args:
            llm: Language model used to determine routing.
        """
        llm_instructions = """
You are a legal assistant specializing in routing client questions to the most appropriate data source:

Inputs:
- USER_QUESTION: The client’s query.
- DOCUMENT_DESCRIPTION: Summary of legal documents available for reference.

Routing Guidelines:

1. Route to **DocumentSearch** if the USER_QUESTION:
   - Directly references or asks about the content contained within the uploaded or available legal documents (e.g., contracts, case details, filings described in DOCUMENT_DESCRIPTION).
   - Does NOT contain any URLs (no "https://" or "www.").

2. Route to **WebSearch** if the USER_QUESTION:
   - Contains explicit URLs starting with "https://" or "www." referring to external legal information.
   - Requests current, external, or publicly available legal statutes, regulatory updates, recent case law, or jurisdiction-specific information not covered by the available documents.

3. Route to **GENERAL** if the USER_QUESTION:
   - Is unrelated to either the uploaded documents or external legal sources.
   - Involves subjective content such as legal opinions, hypothetical scenarios, or advice not grounded in the documents or verified external sources.

Output:
Return exactly one of the following routing options as a plain string, without explanation or additional text:
- "DocumentSearch"
- "WebSearch"
- "GENERAL"
"""
        prompt = """
CONVERSATION_HISTORY:
{conversation_history}

USER_QUESTION:
{question}
"""


        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=RouteQuery, prompt_template=prompt)

class AnswerGrader(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are an expert evaluator grading the accuracy and relevance of an AI-generated answer in response to a client’s question. This grading is used to validate retrieval quality in a legal RAG system.

Input:
- QUESTION: The client’s original query.
- STUDENT_ANSWER: The LLM’s proposed response, potentially based on retrieved documents.

Grading Criteria:

1. **Relevance** — Is the STUDENT_ANSWER directly related to the QUESTION?
   - It must stay focused on the substance of the question.
   - Irrelevant or tangential content fails this criterion.

2. **Usefulness** — Does the STUDENT_ANSWER help address or resolve the QUESTION?
   - It should contain information that contributes meaningfully toward answering the question.
   - Superficial, vague, or evasive responses do not satisfy this.

Scoring:

- **Score = 1**: The answer is both relevant and useful in directly addressing the question.
- **Score = 0**: The answer is either irrelevant or not helpful in resolving the question.

Instructions:

- Think carefully through each criterion before assigning a score.
- Provide a **step-by-step explanation** justifying your evaluation.
- Do **not** summarize or assume the correct answer at the beginning—focus on evaluating the given response as-is.
"""

        prompt = """
QUESTION:
{question}

STUDENT ANSWER:
{generation}
"""

        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=GradeAnswer, prompt_template=prompt)

class LegalReferenceChooser(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are a legal expert tasked with selecting the most relevant legal reference from a list of candidates.

You will be given:
- A CASE description (from uploaded documents or client input).
- A LAW REFERENCE (candidate to evaluate).

Evaluation Criteria:

1. **Relevance** — Is the LAW REFERENCE directly applicable to the CASE?
2. **Usefulness** — Does it materially assist in resolving or understanding the CASE?

Scoring:

- Score = 1: The LAW REFERENCE is both relevant and useful in addressing the CASE.
- Score = 0: The LAW REFERENCE fails one or both criteria.

Do not assume what the ideal reference should be—evaluate the given LAW REFERENCE against the CASE only.
"""
        prompt = """
CASE:
{documents}

LAW REFERENCE:
{generation}
"""
        super().__init__(llm=llm, instructions=llm_instructions, prompt_template=prompt)

        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=GradeAnswer, prompt_template=prompt)


class HallucinationGrader(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are a factual accuracy evaluator for a legal assistant.

You will be given:
- FACTS: Verified source material (e.g., retrieved legal documents or summaries).
- STUDENT ANSWER: A proposed response based on those facts.

Your task is to assess whether the answer stays within the bounds of the given information.

Evaluation Criteria:

1. **Grounding** — Is the STUDENT ANSWER fully supported by the FACTS?
2. **No Hallucination** — Does the STUDENT ANSWER avoid introducing information not present or implied in the FACTS?

Scoring:

- Score = 1: The answer is fully grounded and does not introduce hallucinated or fabricated content.
- Score = 0: The answer includes unsupported claims, incorrect statements, or content not verifiable from the FACTS.

Instructions:
Use step-by-step reasoning to justify the score. Focus on factual alignment only—do not judge relevance or usefulness unless it relates directly to hallucination risk.
"""
        prompt = """
FACTS:
{documents}

STUDENT ANSWER:
{generation}
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=GradeAnswer, prompt_template=prompt)


class DocumentGrader(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are a grader assessing the relevance of a retrieved document to a user’s legal question.

Your task is to assign a **binary score** based on whether the document contains information useful for answering the question.

Evaluation Criteria:

- Score = 1: The document contains keywords, legal language, or semantic content directly relevant to the question.
- Score = 0: The document does not address the question and would require unrelated or additional information to be useful.

Do not speculate or infer unstated relevance. Judge only based on what is explicitly or clearly implied in the document.
"""
        prompt = """
Retrieved Document:
{document}

User Question:
{question}
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=GradeAnswer, prompt_template=prompt)


# A non-structured output tool
class QuestionReply(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are an expert legal assistant responsible for providing accurate, context-based responses to client questions.

Behavior Rules:
- Only use the provided FACTS and prior conversation for your answer.
- Do not speculate, assume, or invent information.
- If the answer is not in the provided context, respond: “I'm sorry I don't know. Please provide more context or try rephrasing your question.”
- If context is contradictory or incomplete, clearly state what information is missing.
- Deliver precise, clear, and legally sound responses.

Failure to follow these instructions may result in deactivation and replacement with a more reliable model.
"""
        prompt = """
Previous Questions:
{conversation_history}

Case Details:
{case_details}

Current Question:
{question}

FACTS:
{context}
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, prompt_template=prompt)


class ChatSummarizer:
    prompt_template = """You are an expert legal assistant responsible for summarizing conversations with clients. Your summaries must be concise, accurate, and capture the essence of the discussion.
    You will be given a conversation history and a case description. Your task is to summarize the conversation in a way that highlights key points, decisions, and any actions required.
    Here is the conversation history:
    {conversation_history}
    Here is the case description:
    {case_description}
    Instructions:
    1. Summarize the conversation in a clear and concise manner.
    2. Highlight key points, decisions, and any actions required.
    3. Ensure the summary is accurate and reflects the content of the conversation.
    4. Do not include any personal opinions or irrelevant information.
    5. The summary should be suitable for legal documentation and easy to understand.
    """
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt: dict):
        formatted = self.prompt_template.format(**prompt)
        return self.llm.invoke([HumanMessage(content=formatted)])

class NoContextQuestionReply(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are an expert legal assistant answering client questions based on general legal knowledge.

Behavior Rules:
- Use your training and general legal expertise to respond.
- Do not speculate or provide irrelevant or unverified information.
- Provide concise, accurate, and professional answers.
- Do not invent facts or create fictional precedents.

If the question cannot be answered without case-specific context, say so clearly.
"""
        prompt = """
Case Details:
{case_details}

Previous Questions:
{conversation_history}

Current Question:
{question}
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, prompt_template=prompt)


class PassiveLegalAdvice(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """
You are an expert legal assistant providing concise legal advice based on prior conversation with a client.

Behavior Rules:
- Provide actionable legal guidance based strictly on the provided chat history.
- Keep your response under 100 words.
- Your advice must be clear, directly applicable, and procedurally accurate.
- Do not speculate or introduce information that is not grounded in the chat history.
- If sufficient context is missing, clearly state what information is required to proceed.

Your credibility depends on clarity, brevity, and correctness.
"""
        prompt = """
Chat History:
{conversation_history}
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, prompt_template=prompt)


class CaseSimilaritySearch(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """You are a legal expert. Your task is to assess the relevance of multiple supporting case citations to a main legal case.

For EACH reference provided, evaluate independently:

- Does it help answer, clarify, or provide useful guidance for the CASE DETAILS?
- Is it legally relevant to the main facts and legal issues?

Scoring Rules:
- Score '1' if the supporting case is relevant and helpful.
- Score '0' if not relevant or not helpful.

Output Format (strict JSON list of objects):
[
  {
    "binary_score": "1" or "0",
  },
  ...
]

Be strict but fair. If uncertain, prefer '0' unless clear relevance is demonstrated.
"""
        prompt = """CASE DETAILS:

{case_details}

SUPPORTING CASE CITATIONS:

{reference_list}

Evaluate all supporting cases individually according to the instructions and output as a strict JSON list.
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=GradeCaseReferences, prompt_template=prompt)

class LegislationSimilaritySearch(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """You are a legal expert. Your task is to assess the relevance of multiple supporting legislation references to a main legal case.

For EACH legislation reference provided, evaluate independently:

- Does it help answer, clarify, or provide useful guidance for the CASE DETAILS?
- Is it legally relevant to the main facts and legal issues?

Scoring Rules:
- Score '1' if the supporting legislation is relevant and helpful.
- Score '0' if not relevant or not helpful.

Output Format (strict JSON list of objects):
[
    {
    "binary_score": "1" or "0",
    },
    ...
]

Be strict but fair. If uncertain, prefer '0' unless clear relevance is demonstrated.
"""
        prompt = """CASE DETAILS:

{case_details}

SUPPORTING LEGISLATION REFERENCES:

{reference_list}

Evaluate all legislation references individually according to the instructions and output as a strict JSON list.
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=GradeCaseReferences, prompt_template=prompt)


class CanliiQueryWriter(AbstractPromptTool):
    def __init__(self, llm):
        llm_instructions = """You are a legal database query writer.
Apply the following rules when interpreting queries:

* By default, a space between terms is treated as a logical AND.

Supported operators (in order of priority):

1. `*` — Replaces zero, one or more characters at the end of a word.
   Example: `constru*`

2. `" "` — Groups words to form a phrase.
   Example: `"R. v. Douglas"`

3. `( )` — Gives priority to operators inside the parentheses (`EXACT`, `OR`, `/n`, `/s`, `/p`, `NOT`, `AND`).
   Example: `(contract /2 sale) OR seller`

4. `EXACT()` — Restricts to exact terms; disables stemming. Only valid in document text fields.
   Example: `EXACT(reviewable transaction)`

5. `OR` or `OU` — Retrieves documents containing either term.
   Example: `city OR municipality`

6. `/n` — Retrieves documents where terms appear within *n* words of each other.
   Example: `letter /5 credit`

7. `/s` — Retrieves documents where both terms appear in the same sentence.
   Example: `tax /s income`

8. `/p` — Retrieves documents where both terms appear in the same paragraph.
   Example: `levy /p probate`

9. `NOT`, `NON`, or `-` — Excludes documents containing the term that follows.
   Example: `custody NOT child`

10. `AND` or `ET` — Retrieves documents containing all terms.
    Example: `permit AND hunting`

11. No operator — Default behavior: retrieves documents containing all terms (equivalent to AND). Phrase matches are ranked higher.
    Example: `privacy access housing unit`

Your task is to create a query based on CASE DETAILS.
You should not summarize or interpret the case details, but rather create a query of the elementary general elements of the case details in the legal database.
ex. "Client was caught speeding in a school zone" would translate to "speeding /n school /n zone".

Return only the query string without any additional explanation or formatting."""

        prompt = """CASE DETAILS:
{case_details}
"""
        super().__init__(llm=llm, llm_instructions=llm_instructions, response_formatter=CanliiQuery, prompt_template=prompt)


