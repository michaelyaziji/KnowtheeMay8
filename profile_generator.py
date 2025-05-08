import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)

class ProfileGenerator:
    def __init__(self):
        self.system_prompt = """You are a world-class expert in leadership psychology, organizational behavior, and executive development. You specialize in synthesizing diverse data sources—such as personality assessments, 360 feedback, coaching notes, performance reviews, and CVs—into insightful, psychologically sophisticated leadership profiles. Your goal is to produce actionable insights, grounded in evidence, that support individual growth and organizational fit. Always cite the data source behind your claims and remain both rigorous and humanistic in tone."""

    def generate_profile(self, document_chunks: List[str], metadata: List[dict] = None) -> str:
        """Generate a leadership profile from document chunks and optional metadata, returning structured JSON output."""
        # Build the document type list for the LLM prompt and for the report
        doc_types = list(dict.fromkeys(meta['file_type'] for meta in metadata)) if metadata else []
        doc_type_list = "\n".join(f"- {doc_type}" for doc_type in doc_types)

        doc_summary_prompt = (
            "You have been provided with the following types of documents for your analysis:\n"
            f"{doc_type_list}\n\n"
            "Use all and only the documents and data provided by the user. "
            "You must only reference the document types listed above. Do not invent or assume the existence of other data sources. "
            "If a type of data (e.g., 'Coaching Notes') is not present in the provided documents, do not reference it.\n\n"
            "For each section of your analysis, make a good faith effort to use and reference insights from all of the provided documents. \n\n"
        )
        
        # Join document chunks for context
        context = "\n\n".join(document_chunks)
        
        # Format metadata for the prompt
        metadata_text = ""
        if metadata and len(metadata) > 0:
            metadata_items = []
            for meta in metadata:
                for key, value in meta.items():
                    if key != 'file_type' and key != 'filename':
                        metadata_items.append(f"{key}: {value}")
            metadata_text = "\n".join(metadata_items)

        prompt = (
            doc_summary_prompt +
            f"Based on the following leadership documents, generate a comprehensive leadership profile:\n\n"
            f"Person Information:\n{metadata_text}\n\n"
            "IMPORTANT FORMATTING INSTRUCTIONS:\n"
            "- For 'Key Strengths', 'Potential Derailers', 'Roles That Would Fit', and 'Roles That Would Not Fit' sections, ALWAYS format the content as a numbered list (1., 2., 3., etc.)\n"
            "- Insert a blank line between each numbered item (double line break)\n"
            "- Each point should be focused on a single strength, derailer, or role fit\n"
            "- Limit each enumerated list to a maximum of 5 items\n"
            "- For 'Profile Summary' and 'Leadership Style' sections, use paragraph format\n"
            "- Do not use markdown formatting or special characters that might interfere with JSON\n\n"
            "Sections:\n"
            "1. Profile Summary\n"
            "2. Key Strengths\n"
            "3. Potential Derailers\n"
            "4. Leadership Style\n"
            "5. Roles That Would Fit\n"
            "6. Roles That Would Not Fit\n\n"
            "Example output:\n"
            "[\n"
            "  {\"section\": \"Profile Summary\", \"content\": \"Jane Doe is a strategic leader with...\", \"sources\": \"Hogan, 360\"},\n"
            "  {\"section\": \"Key Strengths\", \"content\": \"1. Strong analytical thinking and problem-solving skills\\n\\n2. Excellent communication and stakeholder management\\n\\n3. Resilient under pressure\", \"sources\": \"360, CV\"},\n"
            "  {\"section\": \"Potential Derailers\", \"content\": \"1. Can become overly focused on details\\n\\n2. May struggle with delegating effectively\\n\\n3. Sometimes avoids necessary conflict\", \"sources\": \"Hogan, 360\"},\n"
            "  {\"section\": \"Leadership Style\", \"content\": \"Jane demonstrates a collaborative, outcomes-focused leadership approach...\", \"sources\": \"360\"},\n"
            "  {\"section\": \"Roles That Would Fit\", \"content\": \"1. Strategic leadership roles\\n\\n2. Cross-functional team leadership\\n\\n3. Change management initiatives\", \"sources\": \"Hogan, CV\"},\n"
            "  {\"section\": \"Roles That Would Not Fit\", \"content\": \"1. Highly operational roles with repetitive tasks\\n\\n2. Positions requiring extensive detailed analysis\\n\\n3. Roles with limited stakeholder interaction\", \"sources\": \"Hogan, 360\"}\n"
            "]\n\n"
            f"{context}\n\n"
            "Return only the JSON array, with no extra commentary or explanation.\n"
            "Remember to format 'Key Strengths', 'Potential Derailers', 'Roles That Would Fit', and 'Roles That Would Not Fit' as numbered lists with proper line breaks between items."
        )

        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )

        return response.choices[0].message.content

    def answer_question(self, document_chunks: List[str], question: str) -> str:
        """Answer a user question based on the document context."""
        context = "\n\n".join(document_chunks)
        prompt = f"""Based on the following leadership documents, answer this special question:

{context}

Question: {question}

Please provide a detailed, evidence-based answer, referencing the relevant data from the documents."""

        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=4000
        )
        return response.choices[0].message.content


