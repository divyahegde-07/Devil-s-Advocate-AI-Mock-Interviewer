"""ChromaDB vector store setup and management for ideal intent comparison."""

from typing import List
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from devils_advocate.config import OPENAI_API_KEY, CHROMA_COLLECTION_NAME


# Ideal answer intents for different interview categories
IDEAL_INTENTS = {
    "behavioral": [
        "Used specific data or metrics to drive decisions",
        "Demonstrated clear ownership and accountability",
        "Showed collaboration with cross-functional teams",
        "Identified and mitigated risks proactively",
        "Learned from failure and adapted approach",
        "Prioritized based on impact and feasibility",
        "Resolved conflict through empathy and communication",
        "Took initiative without being asked",
        "Balanced multiple stakeholder needs",
        "Made difficult tradeoffs with clear reasoning"
    ],
    "product_sense": [
        "Considered user needs and pain points explicitly",
        "Analyzed competitive landscape or alternatives",
        "Defined success metrics before implementation",
        "Broke down problem into smaller components",
        "Questioned assumptions and validated hypotheses",
        "Balanced business goals with user experience",
        "Identified edge cases and potential failure modes",
        "Proposed iterative approach with feedback loops",
        "Considered scalability and long-term implications",
        "Used frameworks or structured thinking"
    ]
}


def create_vector_store() -> Chroma:
    """Create and populate ChromaDB vector store with ideal intents.

    Returns:
        Chroma vector store instance
    """

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Create ephemeral ChromaDB client (in-memory)
    client = chromadb.Client(Settings(
        is_persistent=False,
        anonymized_telemetry=False
    ))

    # Prepare documents for vector store
    documents = []
    metadatas = []

    for category, intents in IDEAL_INTENTS.items():
        for intent in intents:
            documents.append(intent)
            metadatas.append({"category": category})

    # Create vector store
    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=CHROMA_COLLECTION_NAME,
        client=client
    )

    return vector_store


def search_similar_intents(
    vector_store: Chroma,
    user_answer: str,
    k: int = 3
) -> List[tuple]:
    """Search for similar ideal intents.

    Args:
        vector_store: Chroma vector store instance
        user_answer: User's answer text
        k: Number of results to return

    Returns:
        List of (document, score) tuples
    """

    results = vector_store.similarity_search_with_score(user_answer, k=k)
    return results
