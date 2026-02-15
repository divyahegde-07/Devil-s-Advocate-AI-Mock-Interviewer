"""Chain for extracting structured context from job descriptions."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from devils_advocate.models.schemas import JobContext
from devils_advocate.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def create_context_extractor():
    """Create a chain that extracts JobContext from raw JD text."""

    parser = PydanticOutputParser(pydantic_object=JobContext)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing job descriptions. Extract the following information:
- Role title
- Company name
- Company values (e.g., "Customer Obsession", "Bias for Action")
- Required skills (both technical and soft skills)

If any field is not explicitly mentioned, use reasonable inference or leave empty.

{format_instructions}"""),
        ("user", "Job Description:\n\n{jd_text}")
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

    chain = prompt | llm | parser

    return chain


def extract_context(jd_text: str) -> JobContext:
    """Extract structured context from a job description.

    Args:
        jd_text: Raw job description text

    Returns:
        JobContext with extracted information
    """
    chain = create_context_extractor()
    parser = PydanticOutputParser(pydantic_object=JobContext)

    result = chain.invoke({
        "jd_text": jd_text,
        "format_instructions": parser.get_format_instructions()
    })

    return result
