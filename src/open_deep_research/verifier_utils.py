"""Verification utilities implementing FACT evaluation logic for citation verification.

This module provides the core verification functionality used by the Research Verifier node
to validate citations in generated research reports using the FACT methodology:
1. Statement extraction and deduplication
2. Content retrieval via Jina AI Reader API
3. Support judgment for each statement
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_api_key_for_model

logger = logging.getLogger(__name__)

##########################
# Data Classes
##########################

@dataclass
class Citation:
    """Represents an extracted citation from the report."""
    fact: str
    ref_idx: str
    url: str
    
@dataclass
class CitationGroup:
    """Represents a group of citations from the same URL."""
    url: str
    facts: List[str] = field(default_factory=list)
    url_content: Optional[str] = None
    validation_results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class VerificationResult:
    """Contains the results of the verification process."""
    original_report: str
    verified_report: str
    total_citations: int
    verified_citations: int
    removed_citations: int
    citation_details: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def verification_rate(self) -> float:
        """Calculate the verification rate as a percentage."""
        if self.total_citations == 0:
            return 1.0
        return self.verified_citations / self.total_citations


##########################
# Prompt Templates
##########################

EXTRACTION_PROMPT_EN = """You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "The market grew by 25% in 2023 15"
2. A segment of text + [number], for example: "The market grew by 25% in 2023[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "The market grew by 25%[15†L10]"
4. [Citation Source](Citation Link), for example: "According to [Reuters Report](https://reuters.com/article)'s data..."

Please identify **all** instances where references are cited in the main text, and extract (fact, ref_idx, url) triplets. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, include enough context before and after the citation to ensure the fact is complete and understandable, rather than just a simple phrase.
2. If a fact cites multiple references, create separate triplets: (fact, ref_idx_1, url_1) and (fact, ref_idx_2, url_2).
3. For inline URL citations (form 4), set ref_idx to "0".
4. If the main text does not specify the exact location of citations (only lists references at the end), return an empty list.

Return a JSON list format, where each item is a triplet:
[
    {{
        "fact": "Text segment from the original document with sufficient context for verification",
        "ref_idx": "The index of the cited reference",
        "url": "The URL of the cited reference"
    }}
]

Here is the research report:
{report_text}

Output only the JSON list directly, without any explanation or commentary."""


DEDUPLICATION_PROMPT_EN = """You will be given a list of statements citing the same source. Deduplicate them and return a list of indices of unique statements. 

Two statements are considered duplicates only if they express *exactly the same factual claim*. If no duplicates exist, return the complete list of indices.

Return a List of integers, where each item is the index of a unique statement:
[1, 3, 5]

Below is the list of statements to deduplicate:
{statements}

Output only the integer list, without any explanation."""


VALIDATION_PROMPT_EN = """You are a citation verification assistant. Determine whether each statement is 'supported', 'unsupported', or 'unknown' based on the reference content.

First, assess whether the reference contains valid content. If the reference contains no valid information (e.g., "page not found", access denied, or empty content), mark all statements as 'unknown'.

For valid references:
- 'supported': The facts or data in the statement can be found entirely or partially in the reference (numbers can be rounded)
- 'unsupported': None of the facts or data in the statement can be found in the reference
- 'unknown': Cannot determine due to reference issues

Return a JSON list format:
[
    {{
        "idx": 1,
        "result": "supported",
        "reasoning": "Brief explanation of why this is supported/unsupported"
    }}
]

Reference content:
<reference>
{reference}
</reference>

Statements to verify:
<statements>
{statements}
</statements>

Output only the JSON list, without any explanation."""


##########################
# Jina AI Reader API
##########################

class JinaReader:
    """Async client for Jina AI Reader API to fetch webpage content."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            logger.warning("Jina API key not provided. Web scraping will not work.")
    
    async def fetch_url(self, url: str, timeout: int = 15) -> Dict[str, Any]:
        """Fetch and extract content from a URL using Jina AI Reader.
        
        Args:
            url: The URL to fetch content from
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing url, title, description, content, or error
        """
        if not self.api_key:
            return {
                'url': url,
                'content': '',
                'error': 'Jina API key not configured'
            }
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return {
                'url': url,
                'content': '',
                'error': f'Invalid URL format: {url}'
            }
        
        jina_url = f'https://r.jina.ai/{url}'
        headers = {
            "Accept": "application/json",
            'Authorization': self.api_key,
            'X-Timeout': str(timeout * 1000),
            'X-Engine': "default",
            "X-With-Generated-Alt": "true",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    jina_url, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=timeout + 5)
                ) as response:
                    if response.status != 200:
                        return {
                            'url': url,
                            'content': '',
                            'error': f'Jina AI Reader returned status {response.status}'
                        }
                    
                    response_data = await response.json()
                    data = response_data.get('data', {})
                    
                    title = data.get('title', '')
                    description = data.get('description', '')
                    content = data.get('content', '')
                    
                    full_content = f"{title}\n\n{description}\n\n{content}"
                    
                    return {
                        'url': data.get('url', url),
                        'title': title,
                        'description': description,
                        'content': full_content,
                    }
                    
        except asyncio.TimeoutError:
            return {
                'url': url,
                'content': '',
                'error': f'Request timed out after {timeout} seconds'
            }
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return {
                'url': url,
                'content': '',
                'error': str(e)
            }
    
    async def fetch_urls_batch(
        self, 
        urls: List[str], 
        timeout: int = 15,
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple URLs concurrently with rate limiting.
        
        Args:
            urls: List of URLs to fetch
            timeout: Request timeout per URL
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping URLs to their fetch results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(url: str) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                result = await self.fetch_url(url, timeout)
                return url, result
        
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        url_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch fetch error: {result}")
                continue
            url, data = result
            url_results[url] = data
        
        return url_results


##########################
# Verification Logic
##########################

def clean_json_response(response: str) -> str:
    """Clean LLM response to extract valid JSON."""
    # Remove markdown code blocks
    response = response.replace("```json", "").replace("```", "")
    
    # Clean escape characters
    response = response.replace("\\>", ">")
    response = response.replace("\\<", "<")
    response = response.replace("\\+", "+")
    response = response.replace("\\~", "~")
    
    return response.strip()


def remove_inline_urls(text: str) -> str:
    """Remove URLs from [title](url) format, keeping just [title]."""
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    return pattern.sub(r'[\1]', text)


async def extract_citations(
    report_text: str,
    model,
    max_retries: int = 3
) -> List[Citation]:
    """Extract citations from a research report using LLM.
    
    Args:
        report_text: The full text of the research report
        model: The LLM model to use for extraction
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of Citation objects extracted from the report
    """
    prompt = EXTRACTION_PROMPT_EN.format(report_text=report_text)
    
    for attempt in range(max_retries):
        try:
            response = await model.ainvoke([HumanMessage(content=prompt)])
            response_text = clean_json_response(response.content)
            
            citations_data = json.loads(response_text)
            
            citations = []
            for item in citations_data:
                citation = Citation(
                    fact=remove_inline_urls(item.get('fact', '')),
                    ref_idx=str(item.get('ref_idx', '0')),
                    url=item.get('url', '')
                )
                if citation.url and citation.fact:
                    citations.append(citation)
            
            logger.info(f"Extracted {len(citations)} citations from report")
            return citations
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error("Failed to extract citations after all retries")
                return []
        except Exception as e:
            logger.warning(f"Extraction error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return []
    
    return []


async def deduplicate_citations(
    citations: List[Citation],
    model,
    max_retries: int = 3
) -> Dict[str, CitationGroup]:
    """Group citations by URL and deduplicate within groups.
    
    Args:
        citations: List of extracted citations
        model: The LLM model for deduplication
        max_retries: Maximum retry attempts
        
    Returns:
        Dictionary mapping URLs to CitationGroup objects
    """
    # Group citations by URL
    citation_groups: Dict[str, List[Citation]] = {}
    for citation in citations:
        if citation.url not in citation_groups:
            citation_groups[citation.url] = []
        citation_groups[citation.url].append(citation)
    
    # Deduplicate within each group
    deduped_groups: Dict[str, CitationGroup] = {}
    
    for url, group_citations in citation_groups.items():
        if len(group_citations) == 1:
            # No deduplication needed for single citation
            deduped_groups[url] = CitationGroup(
                url=url,
                facts=[group_citations[0].fact]
            )
            continue
        
        # Create numbered list of statements for deduplication
        statements = '\n'.join([
            f'{i+1}. {c.fact}' for i, c in enumerate(group_citations)
        ])
        
        prompt = DEDUPLICATION_PROMPT_EN.format(statements=statements)
        
        deduped_indices = None
        for attempt in range(max_retries):
            try:
                response = await model.ainvoke([HumanMessage(content=prompt)])
                response_text = clean_json_response(response.content)
                deduped_indices = json.loads(response_text)
                
                # Validate indices
                if (not deduped_indices or 
                    0 in deduped_indices or 
                    len(deduped_indices) > len(group_citations)):
                    raise ValueError("Invalid deduplication indices")
                
                break
                
            except Exception as e:
                logger.warning(f"Deduplication error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Fallback: keep all citations
                    deduped_indices = list(range(1, len(group_citations) + 1))
        
        # Build deduplicated group
        deduped_facts = [group_citations[i-1].fact for i in deduped_indices]
        deduped_groups[url] = CitationGroup(
            url=url,
            facts=deduped_facts
        )
    
    total_original = len(citations)
    total_deduped = sum(len(g.facts) for g in deduped_groups.values())
    logger.info(f"Deduplicated {total_original} citations to {total_deduped} unique claims")
    
    return deduped_groups


async def fetch_citation_content(
    citation_groups: Dict[str, CitationGroup],
    jina_reader: JinaReader,
    max_content_length: int = 50000
) -> Dict[str, CitationGroup]:
    """Fetch webpage content for all citation URLs.
    
    Args:
        citation_groups: Dictionary of citation groups by URL
        jina_reader: JinaReader instance for fetching
        max_content_length: Maximum content length to retrieve
        
    Returns:
        Updated citation groups with fetched content
    """
    urls = list(citation_groups.keys())
    
    logger.info(f"Fetching content for {len(urls)} unique URLs")
    
    url_results = await jina_reader.fetch_urls_batch(urls)
    
    for url, group in citation_groups.items():
        result = url_results.get(url, {})
        if 'error' in result:
            group.url_content = f"Fetch failed: {result['error']}"
            logger.warning(f"Failed to fetch {url}: {result['error']}")
        else:
            content = result.get('content', '')
            group.url_content = content[:max_content_length] if content else "No content available"
    
    return citation_groups


async def validate_citations(
    citation_groups: Dict[str, CitationGroup],
    model,
    max_retries: int = 3
) -> Dict[str, CitationGroup]:
    """Validate each citation against its source content.
    
    Args:
        citation_groups: Dictionary of citation groups with fetched content
        model: The LLM model for validation
        max_retries: Maximum retry attempts
        
    Returns:
        Updated citation groups with validation results
    """
    async def validate_group(url: str, group: CitationGroup) -> Tuple[str, CitationGroup]:
        """Validate a single citation group."""
        if not group.url_content or group.url_content.startswith("Fetch failed:"):
            # Mark all as unknown if content couldn't be fetched
            group.validation_results = [
                {"idx": i, "result": "unknown", "reasoning": "Could not fetch source content"}
                for i in range(len(group.facts))
            ]
            return url, group
        
        # Prepare statements for validation
        statements = '\n'.join([
            f'{i+1}. {fact}' for i, fact in enumerate(group.facts)
        ])
        
        prompt = VALIDATION_PROMPT_EN.format(
            reference=group.url_content,
            statements=statements
        )
        
        for attempt in range(max_retries):
            try:
                response = await model.ainvoke([HumanMessage(content=prompt)])
                response_text = clean_json_response(response.content)
                validation_results = json.loads(response_text)
                
                # Normalize indices (convert from 1-based to 0-based)
                for result in validation_results:
                    result['idx'] = result.get('idx', 1) - 1
                
                # Verify we have results for all facts
                if len(validation_results) != len(group.facts):
                    raise ValueError(f"Expected {len(group.facts)} results, got {len(validation_results)}")
                
                group.validation_results = validation_results
                return url, group
                
            except Exception as e:
                logger.warning(f"Validation error for {url} on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Fallback: mark all as unknown
                    group.validation_results = [
                        {"idx": i, "result": "unknown", "reasoning": f"Validation failed: {str(e)}"}
                        for i in range(len(group.facts))
                    ]
        
        return url, group
    
    # Validate all groups concurrently
    tasks = [validate_group(url, group) for url, group in citation_groups.items()]
    results = await asyncio.gather(*tasks)
    
    validated_groups = {url: group for url, group in results}
    
    # Log validation statistics
    total_validated = 0
    total_supported = 0
    for group in validated_groups.values():
        for result in group.validation_results:
            if result.get('result') != 'unknown':
                total_validated += 1
                if result.get('result') == 'supported':
                    total_supported += 1
    
    logger.info(f"Validated {total_validated} citations, {total_supported} supported")
    
    return validated_groups


def get_supported_facts(citation_groups: Dict[str, CitationGroup]) -> Dict[str, List[str]]:
    """Extract only the supported facts from validated citation groups.
    
    Args:
        citation_groups: Dictionary of validated citation groups
        
    Returns:
        Dictionary mapping URLs to lists of supported facts
    """
    supported = {}
    
    for url, group in citation_groups.items():
        supported_facts = []
        for result in group.validation_results:
            idx = result.get('idx', 0)
            if result.get('result') == 'supported' and idx < len(group.facts):
                supported_facts.append(group.facts[idx])
        
        if supported_facts:
            supported[url] = supported_facts
    
    return supported


def reconstruct_report_with_verified_citations(
    original_report: str,
    citation_groups: Dict[str, CitationGroup]
) -> str:
    """Reconstruct the report keeping only verified citations.
    
    This function identifies unsupported citations and removes them from the report
    while preserving the overall structure and wording as much as possible.
    
    Args:
        original_report: The original research report
        citation_groups: Dictionary of validated citation groups
        
    Returns:
        The reconstructed report with only verified citations
    """
    verified_report = original_report
    
    # Build a set of unsupported facts (normalized for matching)
    unsupported_facts = set()
    for url, group in citation_groups.items():
        for result in group.validation_results:
            idx = result.get('idx', 0)
            if result.get('result') == 'unsupported' and idx < len(group.facts):
                # Normalize the fact for matching
                fact = group.facts[idx].strip()
                unsupported_facts.add(fact)
    
    if not unsupported_facts:
        logger.info("No unsupported citations found, report unchanged")
        return original_report
    
    # Strategy: For each unsupported fact, try to find and remove the sentence
    # containing it from the report
    lines = verified_report.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_text = line.strip()
        if not line_text:
            cleaned_lines.append(line)
            continue
        
        # Check if this line contains any unsupported facts
        should_remove = False
        for fact in unsupported_facts:
            # Check for exact or fuzzy match
            if fact in line or _fuzzy_match(fact, line):
                should_remove = True
                logger.debug(f"Removing line containing unsupported fact: {line[:100]}...")
                break
        
        if not should_remove:
            cleaned_lines.append(line)
    
    verified_report = '\n'.join(cleaned_lines)
    
    # Clean up any orphaned citation references
    verified_report = _clean_orphaned_citations(verified_report)
    
    # Clean up excessive whitespace
    verified_report = re.sub(r'\n{3,}', '\n\n', verified_report)
    
    return verified_report.strip()


def _fuzzy_match(fact: str, line: str) -> bool:
    """Check if a fact fuzzy-matches a line in the report.
    
    This handles cases where the extracted fact might have slight variations
    from how it appears in the original report.
    """
    # Normalize both strings
    fact_normalized = re.sub(r'\s+', ' ', fact.lower().strip())
    line_normalized = re.sub(r'\s+', ' ', line.lower().strip())
    
    # Check for significant overlap (at least 80% of words match)
    fact_words = set(fact_normalized.split())
    line_words = set(line_normalized.split())
    
    if len(fact_words) < 5:
        # For short facts, require exact substring match
        return fact_normalized in line_normalized
    
    overlap = len(fact_words & line_words)
    return overlap >= len(fact_words) * 0.8


def _clean_orphaned_citations(report: str) -> str:
    """Remove orphaned citation references that no longer have context.
    
    This handles cases like dangling [1] or [Source](url) references.
    """
    # Pattern for numbered citations like [1], [2], etc.
    # Only remove if they appear alone (not part of a sentence)
    report = re.sub(r'^\s*\[\d+\]\s*$', '', report, flags=re.MULTILINE)
    
    # Remove empty list items
    report = re.sub(r'^\s*[-*]\s*$', '', report, flags=re.MULTILINE)
    
    return report


##########################
# Main Verification Function
##########################

async def verify_report(
    report_text: str,
    config: RunnableConfig,
    max_content_length: int = 50000
) -> VerificationResult:
    """Execute the complete FACT verification pipeline on a research report.
    
    This function implements the full verification workflow:
    1. Extract citations from the report
    2. Deduplicate equivalent claims
    3. Fetch source content via Jina AI Reader
    4. Validate each claim against its source
    5. Reconstruct report with only verified claims
    
    Args:
        report_text: The research report to verify
        config: Runtime configuration for model settings
        max_content_length: Maximum content length to fetch per URL
        
    Returns:
        VerificationResult containing the verified report and statistics
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize the verification model
    model_api_key = get_api_key_for_model(configurable.verification_model, config)
    verification_model = init_chat_model(
        model=configurable.verification_model,
        max_tokens=configurable.verification_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    )
    
    # Initialize Jina Reader
    jina_reader = JinaReader()
    
    # Step 1: Extract citations
    logger.info("Step 1: Extracting citations from report...")
    citations = await extract_citations(report_text, verification_model)
    
    if not citations:
        logger.info("No citations found in report, returning original")
        return VerificationResult(
            original_report=report_text,
            verified_report=report_text,
            total_citations=0,
            verified_citations=0,
            removed_citations=0
        )
    
    # Step 2: Deduplicate citations
    logger.info("Step 2: Deduplicating citations...")
    citation_groups = await deduplicate_citations(citations, verification_model)
    
    # Step 3: Fetch source content
    logger.info("Step 3: Fetching source content...")
    citation_groups = await fetch_citation_content(
        citation_groups, 
        jina_reader, 
        max_content_length
    )
    
    # Step 4: Validate citations
    logger.info("Step 4: Validating citations against sources...")
    citation_groups = await validate_citations(citation_groups, verification_model)
    
    # Step 5: Reconstruct report
    logger.info("Step 5: Reconstructing report with verified citations...")
    verified_report = reconstruct_report_with_verified_citations(
        report_text, 
        citation_groups
    )
    
    # Calculate statistics
    total_citations = sum(len(g.facts) for g in citation_groups.values())
    verified_citations = sum(
        1 for g in citation_groups.values()
        for r in g.validation_results
        if r.get('result') == 'supported'
    )
    removed_citations = total_citations - verified_citations
    
    # Build detailed citation results
    citation_details = []
    for url, group in citation_groups.items():
        for i, (fact, result) in enumerate(zip(group.facts, group.validation_results)):
            citation_details.append({
                'url': url,
                'fact': fact,
                'result': result.get('result', 'unknown'),
                'reasoning': result.get('reasoning', '')
            })
    
    logger.info(
        f"Verification complete: {verified_citations}/{total_citations} citations verified, "
        f"{removed_citations} removed"
    )
    
    return VerificationResult(
        original_report=report_text,
        verified_report=verified_report,
        total_citations=total_citations,
        verified_citations=verified_citations,
        removed_citations=removed_citations,
        citation_details=citation_details
    )
