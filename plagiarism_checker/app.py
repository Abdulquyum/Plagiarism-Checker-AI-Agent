"""Flask JSON-RPC bridge for the Mastra plagiarism agent."""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections import Counter
from math import sqrt
from typing import Any, Dict, List, Optional

import requests
from flask import Flask
from flask_jsonrpc import JSONRPC

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


logger = logging.getLogger(__name__)


class MastraAgentError(RuntimeError):
    """Raised when the Mastra agent responds with an error."""


def _create_ngrams(text: str, n: int = 3) -> Dict[str, int]:
    """Create n-grams from text."""
    words = text.lower().split()
    words = [w for w in words if w]
    ngrams = Counter()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams[ngram] += 1
    return dict(ngrams)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def _calculate_text_similarity(text1: str, text2: str) -> Dict[str, float]:
    """Calculate text similarity using multiple methods."""
    ngrams1 = _create_ngrams(text1, 3)
    ngrams2 = _create_ngrams(text2, 3)
    
    # Get all unique n-grams
    all_ngrams = set(ngrams1.keys()) | set(ngrams2.keys())
    
    # Create vectors for cosine similarity
    vec1 = [ngrams1.get(ngram, 0) for ngram in all_ngrams]
    vec2 = [ngrams2.get(ngram, 0) for ngram in all_ngrams]
    
    # Calculate cosine similarity
    cosine = _cosine_similarity(vec1, vec2)
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = sum(min(ngrams1.get(ngram, 0), ngrams2.get(ngram, 0)) for ngram in all_ngrams)
    union = sum(max(ngrams1.get(ngram, 0), ngrams2.get(ngram, 0)) for ngram in all_ngrams)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Calculate n-gram overlap percentage
    matching_ngrams = sum(1 for ngram in ngrams1.keys() if ngram in ngrams2)
    ngram_overlap = matching_ngrams / len(ngrams1) if ngrams1 else 0.0
    
    # Overall score (weighted average)
    overall_score = cosine * 0.4 + jaccard * 0.3 + ngram_overlap * 0.3
    
    return {
        "cosineSimilarity": cosine,
        "jaccardSimilarity": jaccard,
        "ngramOverlap": ngram_overlap,
        "overallScore": overall_score,
    }


def _find_suspicious_sections(
    text: str, reference_text: str, threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """Find potentially plagiarized sections."""
    suspicious_sections = []
    
    # Split into sentences
    sentences = re.findall(r'[^.!?]+[.!?]+', text) or [text]
    ref_words = reference_text.lower().split()
    
    for sentence in sentences:
        words = sentence.lower().split()
        if len(words) < 5:  # Skip very short sentences
            continue
        
        # Find best matching section in reference
        max_similarity = 0.0
        for j in range(len(ref_words) - len(words) + 1):
            ref_section = " ".join(ref_words[j : j + len(words)])
            similarity = _calculate_text_similarity(sentence, ref_section)["overallScore"]
            if similarity > max_similarity:
                max_similarity = similarity
        
        if max_similarity >= threshold:
            start_index = text.find(sentence)
            suspicious_sections.append({
                "text": sentence.strip(),
                "similarity": max_similarity,
                "startIndex": start_index,
                "endIndex": start_index + len(sentence),
            })
    
    return suspicious_sections


def _run_plagiarism_check_fallback(
    text: str, reference_text: Optional[str] = None, threshold: float = 0.7
) -> Dict[str, Any]:
    """Fallback plagiarism check implementation in Python."""
    if reference_text and reference_text.strip():
        similarity = _calculate_text_similarity(text, reference_text)
        suspicious_sections = _find_suspicious_sections(text, reference_text, threshold)
        is_likely_plagiarized = similarity["overallScore"] >= threshold
        
        recommendations = []
        if is_likely_plagiarized:
            recommendations.append("High similarity detected. Consider rewriting the identified sections.")
            recommendations.append("Ensure proper citation if this is a reference to existing work.")
        elif similarity["overallScore"] >= threshold * 0.8:
            recommendations.append("Moderate similarity detected. Review for proper attribution.")
        else:
            recommendations.append("Text appears original. Continue to ensure proper citation of any sources.")
        
        return {
            "isLikelyPlagiarized": is_likely_plagiarized,
            "confidenceScore": similarity["overallScore"],
            "overallSimilarity": similarity["overallScore"],
            "suspiciousSections": suspicious_sections,
            "analysis": {
                "cosineSimilarity": similarity["cosineSimilarity"],
                "jaccardSimilarity": similarity["jaccardSimilarity"],
                "ngramOverlap": similarity["ngramOverlap"],
            },
            "recommendations": recommendations,
        }
    else:
        # Without reference text, perform pattern-based analysis
        words = text.split()
        unique_words = set(w.lower() for w in words)
        diversity_ratio = len(unique_words) / len(words) if words else 0.0
        
        # Check for suspicious patterns
        suspicious_patterns = [
            re.compile(r"(?:citation|source|reference)\s*:\s*\d+", re.IGNORECASE),
            re.compile(r"\[.*?\]", re.IGNORECASE),
        ]
        
        pattern_matches = sum(len(pattern.findall(text)) for pattern in suspicious_patterns)
        
        is_likely_plagiarized = diversity_ratio < 0.3
        confidence_score = 0.6 if is_likely_plagiarized else 0.3
        
        recommendations = []
        if is_likely_plagiarized:
            recommendations.append("Low word diversity detected. Consider rewriting for better originality.")
        if pattern_matches > 0:
            recommendations.append("Citation patterns detected. Ensure all sources are properly attributed.")
        recommendations.append("For accurate plagiarism detection, provide reference text to compare against.")
        
        return {
            "isLikelyPlagiarized": is_likely_plagiarized,
            "confidenceScore": confidence_score,
            "overallSimilarity": 0.0,
            "suspiciousSections": [],
            "recommendations": recommendations,
        }


def _build_default_payload(
    text: str,
    reference_text: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the message payload expected by Mastra's A2A endpoint."""

    parts = [{"kind": "text", "text": text}]
    if reference_text:
        parts.append(
            {
                "kind": "text",
                "text": f"Reference text provided by user:\n{reference_text}",
            }
        )
    
    # Add threshold to message if provided in metadata
    if metadata and "threshold" in metadata:
        threshold = metadata.get("threshold")
        parts.append(
            {
                "kind": "text",
                "text": f"Similarity threshold: {threshold}",
            }
        )

    payload: Dict[str, Any] = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": uuid.uuid4().hex,
                "parts": parts,
            }
        },
    }

    if metadata:
        payload["params"]["metadata"] = metadata

    return payload


def call_plagiarism_agent(
    text: str,
    reference_text: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Send the user's text to the Mastra plagiarism agent via HTTP.

    This implementation assumes you have a running Mastra instance (for example,
    started with ``mastra start``) that exposes an A2A endpoint for the
    plagiarism agent. Configure the endpoint via the ``MASTRA_AGENT_URL``
    environment variable if necessary.
    """

    if not text or not text.strip():
        raise ValueError("Text to analyze must be a non-empty string")

    if load_dotenv:
        load_dotenv()

    agent_url = os.getenv(
        "MASTRA_AGENT_URL",
        "http://127.0.0.1:8787/a2a/plagiarismAgent",
    )

    # Optional metadata forwarded to the agent
    metadata: Dict[str, Any] = {}
    if threshold is not None:
        metadata["threshold"] = threshold

    payload = _build_default_payload(
        text.strip(),
        reference_text,
        metadata if metadata else None,
    )

    try:
        response = requests.post(
            agent_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(5, int(os.getenv("MASTRA_AGENT_TIMEOUT", "60"))),
        )
    except requests.RequestException as exc:  # pragma: no cover - network errors
        logger.exception("Failed to reach Mastra agent")
        raise MastraAgentError("Failed to reach Mastra agent") from exc

    if response.status_code >= 400:
        logger.error(
            "Mastra agent responded with status %s: %s",
            response.status_code,
            response.text,
        )
        raise MastraAgentError(
            f"Mastra agent error (status {response.status_code}): {response.text}"
        )

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected payload
        logger.exception("Mastra agent returned invalid JSON")
        raise MastraAgentError("Mastra agent returned invalid JSON") from exc

app = Flask(__name__)
jsonrpc = JSONRPC(app, '/api')

def _extract_tool_results(agent_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract tool call results from the Mastra response structure."""
    try:
        result = agent_response.get("result", {})
        if not result:
            return None
        
        # Helper to recursively search for tool results in nested structures
        def search_for_tool_results(obj: Any, depth: int = 0) -> Optional[Dict[str, Any]]:
            if depth > 10:  # Prevent infinite recursion
                return None
            
            if isinstance(obj, dict):
                # Check if this dict itself is a tool result
                if "isLikelyPlagiarized" in obj or "confidenceScore" in obj:
                    return obj
                
                # Check common keys where tool results might be stored
                for key in ["data", "result", "output", "toolResult", "tool_result", "response"]:
                    if key in obj:
                        found = search_for_tool_results(obj[key], depth + 1)
                        if found:
                            return found
                
                # Recursively search all values
                for value in obj.values():
                    found = search_for_tool_results(value, depth + 1)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = search_for_tool_results(item, depth + 1)
                    if found:
                        return found
            
            return None
        
        # Search the entire response structure
        tool_results = search_for_tool_results(result)
        if tool_results:
            return tool_results
        
        # Also check artifacts explicitly
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            found = search_for_tool_results(artifact)
            if found:
                return found
        
        # Check history for tool calls/results
        history = result.get("history", [])
        for item in history:
            found = search_for_tool_results(item)
            if found:
                return found
        
        return None
    except Exception as exc:
        logger.exception("Error extracting tool results")
        return None


def _format_plagiarism_response(tool_results: Dict[str, Any]) -> str:
    """Format tool results into a readable plagiarism analysis report."""
    try:
        is_plagiarized = tool_results.get("isLikelyPlagiarized", False)
        confidence = tool_results.get("confidenceScore", 0.0)
        overall_similarity = tool_results.get("overallSimilarity", 0.0)
        analysis = tool_results.get("analysis", {})
        suspicious_sections = tool_results.get("suspiciousSections", [])
        recommendations = tool_results.get("recommendations", [])
        
        status = "Plagiarism Detected" if is_plagiarized else "No Plagiarism Detected"
        
        report = f"""## Plagiarism Analysis Results

**Status:** {status}

**Confidence Score:** {confidence:.2f} ({confidence*100:.1f}%)
**Overall Similarity:** {overall_similarity:.2f} ({overall_similarity*100:.1f}%)

**Similarity Metrics:**"""
        
        if analysis:
            cosine = analysis.get("cosineSimilarity")
            jaccard = analysis.get("jaccardSimilarity")
            ngram = analysis.get("ngramOverlap")
            
            if cosine is not None:
                report += f"\n- Cosine Similarity: {cosine:.3f}"
            if jaccard is not None:
                report += f"\n- Jaccard Similarity: {jaccard:.3f}"
            if ngram is not None:
                report += f"\n- N-gram Overlap: {ngram:.3f}"
        else:
            report += "\n- Similarity metrics not available"
        
        report += "\n\n**Suspicious Sections:**"
        if suspicious_sections:
            for i, section in enumerate(suspicious_sections, 1):
                section_text = section.get("text", "")[:100]  # Limit length
                similarity = section.get("similarity", 0.0)
                report += f"\n{i}. Similarity: {similarity:.2f} - {section_text}..."
        else:
            report += "\nNone identified"
        
        report += "\n\n**Recommendations:**"
        if recommendations:
            for rec in recommendations:
                report += f"\n- {rec}"
        else:
            report += "\n- No specific recommendations"
        
        report += f"\n\n**Conclusion:**\n"
        if is_plagiarized:
            report += f"The text shows significant similarity ({overall_similarity*100:.1f}%) indicating potential plagiarism. "
        else:
            report += f"The text shows low similarity ({overall_similarity*100:.1f}%) and appears to be original. "
        report += f"Confidence level: {confidence*100:.1f}%."
        
        return report
    except Exception as exc:
        logger.exception("Error formatting plagiarism response")
        return f"Error formatting results: {exc}"


def _extract_agent_response(agent_response: Dict[str, Any]) -> Optional[str]:
    """Extract the agent's text response from the Mastra response structure."""
    try:
        # Navigate through the response structure to find agent message
        result = agent_response.get("result", {})
        if not result:
            return None
        
        # Check status message (latest response)
        status = result.get("status", {})
        if status:
            message = status.get("message", {})
            if message and message.get("role") == "agent":
                parts = message.get("parts", [])
                # Extract text from all text parts
                texts = [
                    part.get("text", "")
                    for part in parts
                    if part.get("kind") == "text" and part.get("text", "").strip()
                ]
                if texts:
                    return "\n".join(texts)
        
        # Check history for agent messages (reverse to get latest first)
        history = result.get("history", [])
        for item in reversed(history):
            if item.get("role") == "agent" and item.get("kind") == "message":
                parts = item.get("parts", [])
                texts = [
                    part.get("text", "")
                    for part in parts
                    if part.get("kind") == "text" and part.get("text", "").strip()
                ]
                if texts:
                    return "\n".join(texts)
        
        return None
    except Exception as exc:
        logger.exception("Error extracting agent response")
        return None


@jsonrpc.method('plagiarism.check')
def check_plagiarism(
    text: str,
    reference_text: Optional[str] = None,
    threshold: Optional[float] = None,
) -> dict:
    """
    This function is called via JSON-RPC 2.0.
    'text' is the user's input from Telex.im.
    """
    try:
        agent_response = call_plagiarism_agent(
            text=text,
            reference_text=reference_text,
            threshold=threshold,
        )
        
        # Extract the agent's text response
        agent_text = _extract_agent_response(agent_response)
        
        # If agent didn't generate text, try to extract tool results and format them
        if not agent_text or not agent_text.strip():
            logger.warning("Agent returned empty response, attempting to extract tool results")
            tool_results = _extract_tool_results(agent_response)
            
            if tool_results:
                logger.info("Tool results found, generating formatted response")
                agent_text = _format_plagiarism_response(tool_results)
                return {
                    "result": "success",
                    "data": agent_response,
                    "message": "Plagiarism check completed (response generated from tool results).",
                    "agent_response": agent_text,
                }
            else:
                # Fallback: Run plagiarism check directly in Python
                logger.warning(
                    "Agent returned empty response and no tool results found. "
                    "Running fallback plagiarism check."
                )
                fallback_threshold = threshold if threshold is not None else 0.7
                tool_results = _run_plagiarism_check_fallback(
                    text, reference_text, fallback_threshold
                )
                agent_text = _format_plagiarism_response(tool_results)
                return {
                    "result": "success",
                    "data": agent_response,
                    "message": "Plagiarism check completed using fallback implementation (agent did not return results).",
                    "agent_response": agent_text,
                }
        
        return {
            "result": "success",
            "data": agent_response,
            "message": "Plagiarism check completed.",
            "agent_response": agent_text,
        }
    except (ValueError, MastraAgentError) as exc:
        return {
            "result": "error",
            "data": None,
            "message": str(exc),
        }
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error while checking plagiarism")
        return {
            "result": "error",
            "data": None,
            "message": f"An unexpected error occurred: {exc}",
        }

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', '3000'))
    app.run(debug=True, host='0.0.0.0', port=port)