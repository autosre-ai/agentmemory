"""Rule-based entity and relation extraction from text."""

import re
from typing import Callable
from .models import EntityType, EntityMention, RelationMention


class EntityExtractor:
    """
    Rule-based entity extractor using regex patterns.
    
    Extracts entities of various types (people, organizations, locations, etc.)
    from text without requiring an LLM.
    """
    
    # Common name prefixes/titles that indicate a person
    PERSON_TITLES = r"(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Sir|Dame|Lord|Lady|President|CEO|CTO|CFO|Director|Manager|Senator|Governor|Mayor|Captain|General|Admiral)"
    
    # Patterns for different entity types
    PATTERNS = {
        # Person: Capitalized names with optional title
        EntityType.PERSON: [
            # Title + Name
            rf"({PERSON_TITLES}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            # Two or more capitalized words (likely a name)
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        ],
        
        # Organization: Companies, institutions, etc.
        EntityType.ORGANIZATION: [
            # Company suffixes
            r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Company|Co\.?|Corporation|Group|Foundation|Institute|University|College|Association|Organization|Agency|Department|Ministry|Commission))\b",
            # Acronym organizations (3+ capital letters)
            r"\b([A-Z]{3,})\b",
            # The + Name + Organization type
            r"\b(The\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Company|Corporation|Foundation|Institute|Organization|Agency|Commission))\b",
        ],
        
        # Location: Cities, countries, places
        EntityType.LOCATION: [
            # Places with geographic suffixes
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|State|Country|County|Province|District|Region|Island|Mountain|River|Lake|Ocean|Sea|Valley|Desert|Forest|Park))\b",
            # Cardinal direction + place
            r"\b((?:North|South|East|West|Northern|Southern|Eastern|Western)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            # In/at/from + Capitalized (location indicator)
            r"(?:in|at|from|to|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ],
        
        # Date: Various date formats
        EntityType.DATE: [
            # Month DD, YYYY
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
            # DD Month YYYY
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
            # YYYY-MM-DD or YYYY/MM/DD
            r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b",
            # Month YYYY
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        ],
        
        # Event: Named events
        EntityType.EVENT: [
            # The + Name + Event type
            r"\b(The\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Conference|Summit|Meeting|Convention|Festival|Games|Olympics|Championship|Tournament|War|Battle|Revolution|Crisis))\b",
            # World + Event
            r"\b(World\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ],
        
        # Product: Product names
        EntityType.PRODUCT: [
            # Brand patterns (CamelCase or Brand + Product)
            r"\b([A-Z][a-z]+[A-Z][A-Za-z]*(?:\s+\d+)?)\b",  # CamelCase like iPhone, MacBook
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+\d+)?(?:\s+(?:Pro|Plus|Max|Mini|Ultra|Lite))?)\b",
        ],
    }
    
    # Words to exclude from entity extraction
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
        "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "them", "their", "his", "her", "my", "your", "our",
        "said", "says", "told", "asked", "replied", "answered", "stated",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "today", "tomorrow", "yesterday", "now", "then", "here", "there",
    }
    
    # Common words that look like entities but aren't
    FALSE_POSITIVES = {
        "I", "The", "A", "An", "This", "That", "It", "He", "She", "We", "They",
        "What", "When", "Where", "Why", "How", "Which", "Who", "However", "Therefore",
        "Meanwhile", "Furthermore", "Moreover", "Nevertheless", "Although", "Because",
        "Since", "While", "During", "Before", "After", "Until", "Unless",
    }
    
    def __init__(
        self,
        custom_patterns: dict[EntityType, list[str]] | None = None,
        custom_stopwords: set[str] | None = None,
    ):
        """
        Initialize the entity extractor.
        
        Args:
            custom_patterns: Additional regex patterns for entity types
            custom_stopwords: Additional words to exclude
        """
        self.patterns = dict(self.PATTERNS)
        if custom_patterns:
            for entity_type, patterns in custom_patterns.items():
                if entity_type in self.patterns:
                    self.patterns[entity_type].extend(patterns)
                else:
                    self.patterns[entity_type] = patterns
        
        self.stopwords = self.STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def extract(self, text: str) -> list[EntityMention]:
        """
        Extract entity mentions from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of EntityMention objects
        """
        mentions: list[EntityMention] = []
        seen: set[tuple[str, EntityType]] = set()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, text, re.IGNORECASE if entity_type == EntityType.DATE else 0):
                        entity_text = match.group(1) if match.lastindex else match.group(0)
                        entity_text = entity_text.strip()
                        
                        # Skip stopwords and false positives
                        if entity_text.lower() in self.stopwords:
                            continue
                        if entity_text in self.FALSE_POSITIVES:
                            continue
                        if len(entity_text) < 2:
                            continue
                        
                        # Deduplicate by (lowercase_text, type)
                        key = (entity_text.lower(), entity_type)
                        if key in seen:
                            continue
                        seen.add(key)
                        
                        mention = EntityMention(
                            text=entity_text,
                            entity_type=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7,  # Rule-based confidence
                        )
                        mentions.append(mention)
                except re.error:
                    continue
        
        # Sort by position in text
        mentions.sort(key=lambda m: m.start)
        return mentions
    
    def extract_with_context(
        self, 
        text: str, 
        context_window: int = 50
    ) -> list[tuple[EntityMention, str]]:
        """
        Extract entities with surrounding context.
        
        Args:
            text: Input text
            context_window: Characters of context on each side
            
        Returns:
            List of (EntityMention, context_string) tuples
        """
        mentions = self.extract(text)
        results = []
        
        for mention in mentions:
            start = max(0, mention.start - context_window)
            end = min(len(text), mention.end + context_window)
            context = text[start:end]
            results.append((mention, context))
        
        return results


class RelationExtractor:
    """
    Rule-based relation extractor using pattern matching.
    
    Extracts relations (triples) between entities found in text.
    """
    
    # Relation patterns: (regex pattern, predicate name)
    # Pattern should have two capture groups for subject and object
    RELATION_PATTERNS: list[tuple[str, str]] = [
        # Employment/Affiliation
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works|worked|working)\s+(?:at|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "works_at"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?:CEO|CTO|CFO|President|Director|Manager|Founder|Co-founder|Chairman)\s+of\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "leads"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:joined|joins)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "joined"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:founded|co-founded|started|created|established)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "founded"),
        
        # Location
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:located|based|headquartered)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "located_in"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:lives|lived|residing)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "lives_in"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+born|born)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "born_in"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:from|originally\s+from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "from"),
        
        # Relationships
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:married|is\s+married\s+to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "married_to"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?:son|daughter|child)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "child_of"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?:father|mother|parent)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "parent_of"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?:brother|sister|sibling)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "sibling_of"),
        
        # Ownership/Part-of
        (r"([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:owns|acquired|bought|purchased)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "owns"),
        (r"([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)\s+is\s+(?:a\s+)?(?:part|subsidiary|division)\s+of\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "part_of"),
        (r"([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:merged|partnered)\s+with\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "partnered_with"),
        
        # Knowledge/Creation
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:wrote|authored|published)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "authored"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:invented|discovered|developed|created)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "created"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:studies|studied|researches|researched)\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)", "studies"),
        
        # Generic "is a" / type relations
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an)\s+([a-z]+(?:\s+[a-z]+)*)", "is_a"),
        
        # Communication/Interaction
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:met|meets|met\s+with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "met"),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:knows|knew)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "knows"),
    ]
    
    def __init__(
        self,
        custom_patterns: list[tuple[str, str]] | None = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the relation extractor.
        
        Args:
            custom_patterns: Additional (pattern, predicate) tuples
            min_confidence: Minimum confidence threshold
        """
        self.patterns = list(self.RELATION_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        self.min_confidence = min_confidence
    
    def extract(self, text: str) -> list[RelationMention]:
        """
        Extract relation mentions from text.
        
        Args:
            text: Input text
            
        Returns:
            List of RelationMention objects
        """
        relations: list[RelationMention] = []
        seen: set[tuple[str, str, str]] = set()
        
        for pattern, predicate in self.patterns:
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    if match.lastindex and match.lastindex >= 2:
                        subject = match.group(1).strip()
                        obj = match.group(2).strip()
                        
                        # Skip if subject or object is too short
                        if len(subject) < 2 or len(obj) < 2:
                            continue
                        
                        # Deduplicate
                        key = (subject.lower(), predicate, obj.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        
                        relation = RelationMention(
                            subject_text=subject,
                            predicate=predicate,
                            object_text=obj,
                            confidence=0.6,  # Rule-based confidence
                        )
                        relations.append(relation)
            except re.error:
                continue
        
        return relations
    
    def extract_from_entities(
        self,
        text: str,
        entities: list[EntityMention],
    ) -> list[RelationMention]:
        """
        Extract relations between known entities.
        
        This method looks for relations specifically between
        the provided entity mentions.
        
        Args:
            text: Input text
            entities: List of known entity mentions
            
        Returns:
            List of RelationMention objects
        """
        # First, extract all relations
        all_relations = self.extract(text)
        
        # Filter to only include relations between known entities
        entity_names = {e.text.lower() for e in entities}
        
        filtered = []
        for rel in all_relations:
            if (rel.subject_text.lower() in entity_names and 
                rel.object_text.lower() in entity_names):
                filtered.append(rel)
        
        return filtered


class CombinedExtractor:
    """Combines entity and relation extraction."""
    
    def __init__(
        self,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
    ):
        """Initialize with optional custom extractors."""
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relation_extractor = relation_extractor or RelationExtractor()
    
    def extract_all(
        self, 
        text: str
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        """
        Extract both entities and relations from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (entity_mentions, relation_mentions)
        """
        entities = self.entity_extractor.extract(text)
        relations = self.relation_extractor.extract(text)
        
        return entities, relations
