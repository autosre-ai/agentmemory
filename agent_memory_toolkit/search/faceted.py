"""Faceted search with advanced filtering capabilities.

This module provides faceted search functionality including:
- Multi-dimensional filtering (tags, dates, sources, confidence)
- Dynamic facet generation and aggregation
- Boolean filter expressions (AND, OR, NOT)
- Range filters for numeric and date fields
- Hierarchical facets
- Filter auto-suggestion based on corpus
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Callable, TypeVar, Generic
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FilterOperator(Enum):
    """Operators for filter conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    EXISTS = "exists"
    IS_NULL = "is_null"
    REGEX = "regex"


class BooleanOperator(Enum):
    """Boolean operators for combining filters."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True
    
    def matches(self, document: dict[str, Any]) -> bool:
        """Check if document matches this condition."""
        doc_value = self._get_nested_value(document, self.field)
        
        # Handle EXISTS and IS_NULL specially
        if self.operator == FilterOperator.EXISTS:
            return doc_value is not None
        
        if self.operator == FilterOperator.IS_NULL:
            return doc_value is None
        
        # If value doesn't exist and we're not checking existence, no match
        if doc_value is None:
            return False
        
        return self._check_condition(doc_value)
    
    def _get_nested_value(self, document: dict[str, Any], field: str) -> Any:
        """Get a potentially nested field value."""
        parts = field.split(".")
        value = document
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if idx < len(value) else None
            else:
                return None
            
            if value is None:
                return None
        
        return value
    
    def _check_condition(self, doc_value: Any) -> bool:
        """Check if the document value satisfies the condition."""
        # String operations
        if isinstance(doc_value, str):
            comp_value = doc_value if self.case_sensitive else doc_value.lower()
            match_value = self.value if self.case_sensitive else str(self.value).lower()
            
            if self.operator == FilterOperator.EQUALS:
                return comp_value == match_value
            elif self.operator == FilterOperator.NOT_EQUALS:
                return comp_value != match_value
            elif self.operator == FilterOperator.CONTAINS:
                return match_value in comp_value
            elif self.operator == FilterOperator.NOT_CONTAINS:
                return match_value not in comp_value
            elif self.operator == FilterOperator.STARTS_WITH:
                return comp_value.startswith(match_value)
            elif self.operator == FilterOperator.ENDS_WITH:
                return comp_value.endswith(match_value)
            elif self.operator == FilterOperator.REGEX:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                return bool(re.search(self.value, doc_value, flags))
            elif self.operator == FilterOperator.IN:
                values = [v.lower() for v in self.value] if not self.case_sensitive else self.value
                return comp_value in values
            elif self.operator == FilterOperator.NOT_IN:
                values = [v.lower() for v in self.value] if not self.case_sensitive else self.value
                return comp_value not in values
        
        # List operations (for tags, etc.)
        elif isinstance(doc_value, list):
            if self.operator == FilterOperator.CONTAINS:
                return self.value in doc_value
            elif self.operator == FilterOperator.NOT_CONTAINS:
                return self.value not in doc_value
            elif self.operator == FilterOperator.IN:
                # Check if any of self.value is in doc_value
                return any(v in doc_value for v in self.value)
            elif self.operator == FilterOperator.NOT_IN:
                return not any(v in doc_value for v in self.value)
        
        # Numeric operations
        elif isinstance(doc_value, (int, float)):
            if self.operator == FilterOperator.EQUALS:
                return doc_value == self.value
            elif self.operator == FilterOperator.NOT_EQUALS:
                return doc_value != self.value
            elif self.operator == FilterOperator.GREATER_THAN:
                return doc_value > self.value
            elif self.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return doc_value >= self.value
            elif self.operator == FilterOperator.LESS_THAN:
                return doc_value < self.value
            elif self.operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return doc_value <= self.value
            elif self.operator == FilterOperator.BETWEEN:
                low, high = self.value
                return low <= doc_value <= high
            elif self.operator == FilterOperator.IN:
                return doc_value in self.value
        
        # Date operations
        elif isinstance(doc_value, (datetime, date, str)):
            parsed_value = self._parse_date(doc_value)
            parsed_filter = self._parse_date(self.value) if not isinstance(self.value, tuple) else None
            
            if parsed_value is None:
                return False
            
            if self.operator == FilterOperator.EQUALS:
                return parsed_value.date() == parsed_filter.date() if parsed_filter else False
            elif self.operator == FilterOperator.GREATER_THAN:
                return parsed_value > parsed_filter if parsed_filter else False
            elif self.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return parsed_value >= parsed_filter if parsed_filter else False
            elif self.operator == FilterOperator.LESS_THAN:
                return parsed_value < parsed_filter if parsed_filter else False
            elif self.operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return parsed_value <= parsed_filter if parsed_filter else False
            elif self.operator == FilterOperator.BETWEEN:
                low, high = self.value
                low_dt = self._parse_date(low)
                high_dt = self._parse_date(high)
                return low_dt <= parsed_value <= high_dt if low_dt and high_dt else False
        
        # Generic equality
        if self.operator == FilterOperator.EQUALS:
            return doc_value == self.value
        elif self.operator == FilterOperator.NOT_EQUALS:
            return doc_value != self.value
        
        return False
    
    def _parse_date(self, value: Any) -> datetime | None:
        """Parse a date value to datetime."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        elif isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None


@dataclass
class FilterGroup:
    """A group of filter conditions combined with boolean operators."""
    operator: BooleanOperator = BooleanOperator.AND
    conditions: list["FilterCondition | FilterGroup"] = field(default_factory=list)
    
    def matches(self, document: dict[str, Any]) -> bool:
        """Check if document matches this filter group."""
        if not self.conditions:
            return True
        
        if self.operator == BooleanOperator.AND:
            return all(c.matches(document) for c in self.conditions)
        elif self.operator == BooleanOperator.OR:
            return any(c.matches(document) for c in self.conditions)
        elif self.operator == BooleanOperator.NOT:
            # NOT applies to first condition only
            return not self.conditions[0].matches(document) if self.conditions else True
        
        return False
    
    def add(
        self, 
        field: str, 
        operator: FilterOperator, 
        value: Any,
        case_sensitive: bool = True,
    ) -> "FilterGroup":
        """Add a condition to this group."""
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value,
            case_sensitive=case_sensitive,
        )
        self.conditions.append(condition)
        return self
    
    def add_group(self, group: "FilterGroup") -> "FilterGroup":
        """Add a nested filter group."""
        self.conditions.append(group)
        return self


@dataclass
class FacetValue:
    """A single facet value with count."""
    value: Any
    count: int
    selected: bool = False
    
    def __hash__(self):
        return hash(str(self.value))


@dataclass
class Facet:
    """A facet with its values and aggregations."""
    field: str
    display_name: str
    values: list[FacetValue] = field(default_factory=list)
    total_count: int = 0
    facet_type: str = "terms"  # "terms", "range", "date_histogram"
    
    def top_values(self, n: int = 10) -> list[FacetValue]:
        """Get top N facet values by count."""
        sorted_values = sorted(self.values, key=lambda v: v.count, reverse=True)
        return sorted_values[:n]


@dataclass
class FacetedSearchConfig:
    """Configuration for faceted search."""
    
    # Facet settings
    facet_fields: list[str] = field(default_factory=lambda: ["metadata.tags", "metadata.source"])
    max_facet_values: int = 100
    min_facet_count: int = 1
    
    # Date facets
    date_facet_fields: list[str] = field(default_factory=lambda: ["created_at"])
    date_granularity: str = "day"  # "hour", "day", "week", "month", "year"
    
    # Numeric range facets
    range_facet_fields: list[str] = field(default_factory=lambda: ["metadata.confidence"])
    range_buckets: int = 10
    
    # Filter behavior
    filter_mode: str = "post"  # "pre" filters before scoring, "post" filters after
    
    # Result settings
    include_facets: bool = True
    include_zero_count_facets: bool = False


@dataclass
class FacetedSearchResult:
    """Result of a faceted search."""
    matches: list[dict[str, Any]]
    facets: dict[str, Facet]
    total_count: int
    filtered_count: int
    applied_filters: FilterGroup | None = None


class FacetExtractor:
    """Extracts and aggregates facet values from documents."""
    
    def __init__(self, config: FacetedSearchConfig | None = None):
        self.config = config or FacetedSearchConfig()
    
    def extract_facets(
        self,
        documents: list[dict[str, Any]],
        facet_fields: list[str] | None = None,
    ) -> dict[str, Facet]:
        """
        Extract facets from a list of documents.
        
        Args:
            documents: Documents to extract facets from
            facet_fields: Fields to extract facets for
            
        Returns:
            Dictionary mapping field names to Facet objects
        """
        facet_fields = facet_fields or self.config.facet_fields
        facets: dict[str, Facet] = {}
        
        for field_name in facet_fields:
            facet = self._extract_terms_facet(documents, field_name)
            facets[field_name] = facet
        
        # Extract date facets
        for field_name in self.config.date_facet_fields:
            if field_name not in facets:
                facet = self._extract_date_facet(documents, field_name)
                facets[field_name] = facet
        
        # Extract range facets
        for field_name in self.config.range_facet_fields:
            if field_name not in facets:
                facet = self._extract_range_facet(documents, field_name)
                facets[field_name] = facet
        
        return facets
    
    def _get_nested_value(self, document: dict[str, Any], field: str) -> Any:
        """Get a potentially nested field value."""
        parts = field.split(".")
        value = document
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if idx < len(value) else None
            else:
                return None
            
            if value is None:
                return None
        
        return value
    
    def _extract_terms_facet(
        self, 
        documents: list[dict[str, Any]], 
        field_name: str,
    ) -> Facet:
        """Extract a terms facet for a field."""
        value_counts: dict[Any, int] = defaultdict(int)
        
        for doc in documents:
            value = self._get_nested_value(doc, field_name)
            
            if value is None:
                continue
            
            # Handle lists (like tags)
            if isinstance(value, list):
                for item in value:
                    if item is not None:
                        value_counts[item] += 1
            else:
                value_counts[value] += 1
        
        # Convert to FacetValues
        facet_values = []
        for value, count in value_counts.items():
            if count >= self.config.min_facet_count:
                facet_values.append(FacetValue(value=value, count=count))
        
        # Sort by count descending
        facet_values.sort(key=lambda v: v.count, reverse=True)
        
        # Limit values
        facet_values = facet_values[:self.config.max_facet_values]
        
        return Facet(
            field=field_name,
            display_name=self._field_to_display_name(field_name),
            values=facet_values,
            total_count=len(documents),
            facet_type="terms",
        )
    
    def _extract_date_facet(
        self, 
        documents: list[dict[str, Any]], 
        field_name: str,
    ) -> Facet:
        """Extract a date histogram facet."""
        date_counts: dict[str, int] = defaultdict(int)
        
        for doc in documents:
            value = self._get_nested_value(doc, field_name)
            
            if value is None:
                continue
            
            # Parse date
            if isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
            elif isinstance(value, datetime):
                dt = value
            elif isinstance(value, date):
                dt = datetime.combine(value, datetime.min.time())
            else:
                continue
            
            # Truncate to granularity
            bucket_key = self._date_to_bucket(dt)
            date_counts[bucket_key] += 1
        
        # Convert to FacetValues
        facet_values = []
        for bucket_key, count in sorted(date_counts.items()):
            facet_values.append(FacetValue(value=bucket_key, count=count))
        
        return Facet(
            field=field_name,
            display_name=self._field_to_display_name(field_name),
            values=facet_values,
            total_count=len(documents),
            facet_type="date_histogram",
        )
    
    def _date_to_bucket(self, dt: datetime) -> str:
        """Convert datetime to bucket key based on granularity."""
        if self.config.date_granularity == "hour":
            return dt.strftime("%Y-%m-%d %H:00")
        elif self.config.date_granularity == "day":
            return dt.strftime("%Y-%m-%d")
        elif self.config.date_granularity == "week":
            # Start of week (Monday)
            start = dt - timedelta(days=dt.weekday())
            return start.strftime("%Y-%m-%d")
        elif self.config.date_granularity == "month":
            return dt.strftime("%Y-%m")
        elif self.config.date_granularity == "year":
            return dt.strftime("%Y")
        else:
            return dt.strftime("%Y-%m-%d")
    
    def _extract_range_facet(
        self, 
        documents: list[dict[str, Any]], 
        field_name: str,
    ) -> Facet:
        """Extract a numeric range facet."""
        values: list[float] = []
        
        for doc in documents:
            value = self._get_nested_value(doc, field_name)
            
            if value is not None and isinstance(value, (int, float)):
                values.append(float(value))
        
        if not values:
            return Facet(
                field=field_name,
                display_name=self._field_to_display_name(field_name),
                values=[],
                total_count=len(documents),
                facet_type="range",
            )
        
        # Create buckets
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            # Single value
            facet_values = [FacetValue(value=f"{min_val}", count=len(values))]
        else:
            bucket_size = (max_val - min_val) / self.config.range_buckets
            bucket_counts: dict[str, int] = defaultdict(int)
            
            for val in values:
                bucket_idx = min(
                    int((val - min_val) / bucket_size),
                    self.config.range_buckets - 1
                )
                bucket_start = min_val + bucket_idx * bucket_size
                bucket_end = min_val + (bucket_idx + 1) * bucket_size
                bucket_key = f"{bucket_start:.2f}-{bucket_end:.2f}"
                bucket_counts[bucket_key] += 1
            
            facet_values = [
                FacetValue(value=key, count=count)
                for key, count in sorted(bucket_counts.items())
            ]
        
        return Facet(
            field=field_name,
            display_name=self._field_to_display_name(field_name),
            values=facet_values,
            total_count=len(documents),
            facet_type="range",
        )
    
    def _field_to_display_name(self, field_name: str) -> str:
        """Convert field name to display name."""
        # Remove metadata prefix if present
        if field_name.startswith("metadata."):
            field_name = field_name[9:]
        
        # Convert snake_case to Title Case
        return field_name.replace("_", " ").title()


class FilterBuilder:
    """Fluent builder for constructing filter expressions."""
    
    def __init__(self):
        self._root = FilterGroup(operator=BooleanOperator.AND)
        self._current_group = self._root
    
    def where(
        self, 
        field: str, 
        operator: FilterOperator | str, 
        value: Any,
        case_sensitive: bool = True,
    ) -> "FilterBuilder":
        """Add a filter condition."""
        if isinstance(operator, str):
            operator = FilterOperator(operator)
        
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value,
            case_sensitive=case_sensitive,
        )
        self._current_group.conditions.append(condition)
        return self
    
    def equals(self, field: str, value: Any) -> "FilterBuilder":
        """Add equality filter."""
        return self.where(field, FilterOperator.EQUALS, value)
    
    def not_equals(self, field: str, value: Any) -> "FilterBuilder":
        """Add not-equals filter."""
        return self.where(field, FilterOperator.NOT_EQUALS, value)
    
    def contains(self, field: str, value: Any) -> "FilterBuilder":
        """Add contains filter."""
        return self.where(field, FilterOperator.CONTAINS, value)
    
    def in_list(self, field: str, values: list[Any]) -> "FilterBuilder":
        """Add IN filter."""
        return self.where(field, FilterOperator.IN, values)
    
    def not_in_list(self, field: str, values: list[Any]) -> "FilterBuilder":
        """Add NOT IN filter."""
        return self.where(field, FilterOperator.NOT_IN, values)
    
    def greater_than(self, field: str, value: Any) -> "FilterBuilder":
        """Add greater than filter."""
        return self.where(field, FilterOperator.GREATER_THAN, value)
    
    def greater_than_or_equal(self, field: str, value: Any) -> "FilterBuilder":
        """Add greater than or equal filter."""
        return self.where(field, FilterOperator.GREATER_THAN_OR_EQUAL, value)
    
    def less_than(self, field: str, value: Any) -> "FilterBuilder":
        """Add less than filter."""
        return self.where(field, FilterOperator.LESS_THAN, value)
    
    def less_than_or_equal(self, field: str, value: Any) -> "FilterBuilder":
        """Add less than or equal filter."""
        return self.where(field, FilterOperator.LESS_THAN_OR_EQUAL, value)
    
    def between(self, field: str, low: Any, high: Any) -> "FilterBuilder":
        """Add between filter."""
        return self.where(field, FilterOperator.BETWEEN, (low, high))
    
    def exists(self, field: str) -> "FilterBuilder":
        """Add exists filter."""
        return self.where(field, FilterOperator.EXISTS, True)
    
    def is_null(self, field: str) -> "FilterBuilder":
        """Add is null filter."""
        return self.where(field, FilterOperator.IS_NULL, True)
    
    def regex(self, field: str, pattern: str, case_sensitive: bool = True) -> "FilterBuilder":
        """Add regex filter."""
        return self.where(field, FilterOperator.REGEX, pattern, case_sensitive)
    
    def and_group(self) -> "FilterBuilder":
        """Start an AND group."""
        new_group = FilterGroup(operator=BooleanOperator.AND)
        self._current_group.conditions.append(new_group)
        self._current_group = new_group
        return self
    
    def or_group(self) -> "FilterBuilder":
        """Start an OR group."""
        new_group = FilterGroup(operator=BooleanOperator.OR)
        self._current_group.conditions.append(new_group)
        self._current_group = new_group
        return self
    
    def end_group(self) -> "FilterBuilder":
        """End current group (return to parent)."""
        self._current_group = self._root
        return self
    
    def has_tag(self, tag: str) -> "FilterBuilder":
        """Convenience method to filter by tag."""
        return self.contains("metadata.tags", tag)
    
    def has_any_tag(self, tags: list[str]) -> "FilterBuilder":
        """Convenience method to filter by any of multiple tags."""
        return self.where("metadata.tags", FilterOperator.IN, tags)
    
    def from_source(self, source: str) -> "FilterBuilder":
        """Convenience method to filter by source."""
        return self.equals("metadata.source", source)
    
    def min_confidence(self, confidence: float) -> "FilterBuilder":
        """Convenience method to filter by minimum confidence."""
        return self.greater_than_or_equal("metadata.confidence", confidence)
    
    def created_after(self, dt: datetime | str) -> "FilterBuilder":
        """Filter to documents created after a date."""
        return self.greater_than("created_at", dt)
    
    def created_before(self, dt: datetime | str) -> "FilterBuilder":
        """Filter to documents created before a date."""
        return self.less_than("created_at", dt)
    
    def created_between(self, start: datetime | str, end: datetime | str) -> "FilterBuilder":
        """Filter to documents created between two dates."""
        return self.between("created_at", start, end)
    
    def build(self) -> FilterGroup:
        """Build and return the filter group."""
        return self._root


class FacetedSearchEngine:
    """
    Search engine with faceted filtering capabilities.
    
    Provides:
    - Dynamic facet extraction from document corpus
    - Multi-dimensional filtering
    - Boolean filter expressions
    - Range and date filters
    - Facet value aggregation with counts
    
    Example:
        >>> engine = FacetedSearchEngine()
        >>> 
        >>> # Build filters
        >>> filters = (FilterBuilder()
        ...     .has_tag("important")
        ...     .min_confidence(0.8)
        ...     .created_after("2024-01-01")
        ...     .build()
        ... )
        >>> 
        >>> # Search with filters
        >>> results = engine.search(
        ...     documents=memories,
        ...     filters=filters,
        ...     include_facets=True,
        ... )
        >>> 
        >>> # Use facets for refinement
        >>> print(results.facets["metadata.tags"].top_values(5))
    """
    
    def __init__(self, config: FacetedSearchConfig | None = None):
        self.config = config or FacetedSearchConfig()
        self.facet_extractor = FacetExtractor(config=self.config)
    
    def search(
        self,
        documents: list[dict[str, Any]],
        filters: FilterGroup | None = None,
        query: str | None = None,
        content_field: str = "content",
        include_facets: bool | None = None,
        facet_fields: list[str] | None = None,
    ) -> FacetedSearchResult:
        """
        Search documents with faceted filtering.
        
        Args:
            documents: Documents to search
            filters: Filter conditions to apply
            query: Optional text query (simple substring match)
            content_field: Field to search for query
            include_facets: Whether to include facet aggregations
            facet_fields: Override default facet fields
            
        Returns:
            FacetedSearchResult with matches, facets, and counts
        """
        include_facets = include_facets if include_facets is not None else self.config.include_facets
        total_count = len(documents)
        
        # Apply text query filter if provided
        if query:
            query_lower = query.lower()
            documents = [
                doc for doc in documents
                if query_lower in str(doc.get(content_field, "")).lower()
            ]
        
        # Apply filters
        if filters:
            documents = [doc for doc in documents if filters.matches(doc)]
        
        filtered_count = len(documents)
        
        # Extract facets (from filtered documents)
        facets: dict[str, Facet] = {}
        if include_facets:
            facets = self.facet_extractor.extract_facets(
                documents,
                facet_fields=facet_fields,
            )
        
        return FacetedSearchResult(
            matches=documents,
            facets=facets,
            total_count=total_count,
            filtered_count=filtered_count,
            applied_filters=filters,
        )
    
    def get_filter_suggestions(
        self,
        documents: list[dict[str, Any]],
        current_filters: FilterGroup | None = None,
        top_n: int = 5,
    ) -> dict[str, list[FacetValue]]:
        """
        Get filter value suggestions based on document corpus.
        
        Args:
            documents: Document corpus
            current_filters: Currently applied filters
            top_n: Number of suggestions per field
            
        Returns:
            Dictionary mapping fields to suggested FacetValues
        """
        # Apply current filters to get relevant subset
        if current_filters:
            documents = [doc for doc in documents if current_filters.matches(doc)]
        
        # Extract facets
        facets = self.facet_extractor.extract_facets(documents)
        
        # Get top values for each facet
        suggestions = {}
        for field, facet in facets.items():
            suggestions[field] = facet.top_values(top_n)
        
        return suggestions


def filter_documents(
    documents: list[dict[str, Any]],
    filters: FilterGroup,
) -> list[dict[str, Any]]:
    """
    Convenience function to filter documents.
    
    Args:
        documents: Documents to filter
        filters: Filter conditions
        
    Returns:
        Filtered list of documents
    """
    return [doc for doc in documents if filters.matches(doc)]


def build_filter(
    tags: list[str] | None = None,
    source: str | None = None,
    min_confidence: float | None = None,
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
) -> FilterGroup:
    """
    Convenience function to build common filters.
    
    Args:
        tags: Filter by any of these tags
        source: Filter by source
        min_confidence: Minimum confidence threshold
        created_after: Filter by creation date
        created_before: Filter by creation date
        
    Returns:
        FilterGroup with all conditions
    """
    builder = FilterBuilder()
    
    if tags:
        builder.has_any_tag(tags)
    
    if source:
        builder.from_source(source)
    
    if min_confidence is not None:
        builder.min_confidence(min_confidence)
    
    if created_after:
        builder.created_after(created_after)
    
    if created_before:
        builder.created_before(created_before)
    
    return builder.build()
