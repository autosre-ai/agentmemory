# Temporal Memory Guide

Agent Memory Toolkit provides cognitively-inspired temporal memory capabilities that model how human memory works over time. This guide covers the four main components of the temporal memory system.

## Overview

The temporal module (`agent_memory_toolkit.temporal`) offers:

| Component | Purpose | Cognitive Inspiration |
|-----------|---------|----------------------|
| **Timeline** | Chronological organization | Autobiographical memory |
| **Episodes** | Decay and forgetting | Ebbinghaus forgetting curve |
| **Working Memory** | Limited capacity + attention | Miller's 7±2 chunks |
| **Consolidation** | Short to long-term transfer | Sleep consolidation |

## Quick Start

```python
from datetime import datetime
from agent_memory_toolkit.temporal import (
    Timeline,
    TimeWindow,
    TemporalMemory,
    EpisodicMemoryStore,
    EmotionalValence,
    WorkingMemory,
    MemoryConsolidator,
)

# 1. Timeline organization
timeline = Timeline()
timeline.add(TemporalMemory(
    memory_id="1",
    content="User asked about Python decorators",
    timestamp=datetime.now(),
))

# 2. Episodic memory with decay
episodes = EpisodicMemoryStore()
episodes.add_item(
    content="Decorators wrap functions",
    importance=0.8,
    emotional_valence=EmotionalValence.POSITIVE,
)

# 3. Working memory with limits
wm = WorkingMemory()
wm.add("Current task: Explain decorators", priority=0.9)

# 4. Consolidate to long-term storage
consolidator = MemoryConsolidator(working_memory=wm, episodic_store=episodes)
result = consolidator.consolidate()
```

## Timeline

The Timeline provides chronological organization and time-based retrieval of memories.

### Basic Usage

```python
from datetime import datetime, timedelta
from agent_memory_toolkit.temporal import (
    Timeline,
    TimelineConfig,
    TemporalMemory,
    TimeWindow,
    RecencyDecay,
)

# Configure timeline
config = TimelineConfig(
    cluster_granularity=timedelta(hours=1),
    recency_decay=RecencyDecay.EXPONENTIAL,
    decay_half_life=timedelta(days=7),
)

timeline = Timeline(config)

# Add memories
timeline.add(TemporalMemory(
    memory_id="mem_1",
    content="Started the project discussion",
    timestamp=datetime(2024, 1, 15, 10, 0),
))

timeline.add(TemporalMemory(
    memory_id="mem_2",
    content="Decided on Python for backend",
    timestamp=datetime(2024, 1, 15, 10, 30),
))

# Query by time window
result = timeline.query(time_window=TimeWindow.LAST_DAY)
for memory in result.memories:
    print(f"{memory.timestamp}: {memory.content}")
```

### Time Windows

Predefined time windows for easy retrieval:

```python
from agent_memory_toolkit.temporal import TimeWindow

# Available time windows
TimeWindow.LAST_MINUTE
TimeWindow.LAST_HOUR
TimeWindow.LAST_DAY
TimeWindow.LAST_WEEK
TimeWindow.LAST_MONTH
TimeWindow.LAST_QUARTER
TimeWindow.LAST_YEAR
TimeWindow.ALL_TIME

# Query with time window
recent = timeline.query(time_window=TimeWindow.LAST_HOUR)
```

### Recency Scoring

Apply recency-weighted boosts to search results:

```python
from agent_memory_toolkit.temporal import RecencyDecay

# Configure decay behavior
config = TimelineConfig(
    recency_decay=RecencyDecay.EXPONENTIAL,  # or LINEAR, LOGARITHMIC, STEP
    decay_half_life=timedelta(days=7),
    max_recency_boost=1.0,
)

timeline = Timeline(config)

# Get memories with recency scores
scored = timeline.get_with_recency_scores(memories)
for memory, score in scored:
    print(f"Score {score:.2f}: {memory.content}")

# Boost search results with recency
search_results = [(memory, relevance_score), ...]
boosted = timeline.boost_search_results(
    results=search_results,
    recency_weight=0.3,  # 30% recency, 70% relevance
)
```

### Temporal Clustering

Automatically group related memories by time:

```python
# Query with clustering
result = timeline.query(
    time_window=TimeWindow.LAST_DAY,
    include_clusters=True,
)

# Access clusters
for cluster in result.clusters:
    print(f"Cluster: {cluster.start_time} - {cluster.end_time}")
    print(f"  Items: {cluster.memory_count}")
    for mem in cluster.memories:
        print(f"    - {mem.content}")
```

### Direct Access

```python
# Get by specific time periods
today = timeline.get_by_day(datetime.now())
this_week = timeline.get_by_week(datetime.now())
this_month = timeline.get_by_month(datetime.now())

# Get most recent
recent_10 = timeline.get_recent(n=10)

# Statistics
stats = timeline.get_timeline_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"Span: {stats['span']}")
```

## Episodic Memory

The EpisodicMemoryStore implements memory with decay and forgetting curves, inspired by cognitive psychology research.

### Basic Usage

```python
from agent_memory_toolkit.temporal import (
    EpisodicMemoryStore,
    EpisodicMemoryConfig,
    DecayModel,
    EmotionalValence,
)

# Configure with decay model
config = EpisodicMemoryConfig(
    decay_model=DecayModel.EBBINGHAUS,  # Classic forgetting curve
    forgetting_threshold=0.1,
    enable_forgetting=True,  # Actually remove forgotten memories
)

store = EpisodicMemoryStore(config)

# Add memory items
item = store.add_item(
    content="User prefers dark mode interfaces",
    importance=0.7,
    emotional_valence=EmotionalValence.POSITIVE,
)

# Access strengthens memories
retrieved = store.get_item(item.item_id)

# Explicit rehearsal strengthens more
store.rehearse(item.item_id)
```

### Decay Models

Choose from cognitive research-backed decay models:

```python
from agent_memory_toolkit.temporal import DecayModel

# Ebbinghaus forgetting curve: R = e^(-t/S)
DecayModel.EBBINGHAUS

# Power law of forgetting (more gradual)
DecayModel.POWER_LAW

# Simple exponential decay
DecayModel.EXPONENTIAL

# ACT-R base-level learning equation
DecayModel.ACT_R
```

### Applying Decay

```python
# Apply decay to all memories
forgotten_count = store.apply_decay()
print(f"{forgotten_count} memories below threshold")

# Get memories by strength
strong = store.get_strong_memories(min_strength=0.5, limit=10)

# Get by importance
important = store.get_important_memories(min_importance=0.7)
```

### Episodes

Memories are automatically grouped into episodes:

```python
from agent_memory_toolkit.temporal import EpisodeType

# Add items - automatically grouped into episodes
store.add_item("User: Hello", episode_type=EpisodeType.INTERACTION)
store.add_item("Agent: Hi there!", episode_type=EpisodeType.INTERACTION)

# Gap > 30 minutes starts new episode
import time
time.sleep(1)  # In real use, natural gaps form episodes

# Get episodes
episodes = store.get_episodes(min_strength=0.3)
for episode in episodes:
    print(f"Episode: {episode.episode_type.value}")
    print(f"  Duration: {episode.duration}")
    print(f"  Items: {episode.item_count}")

# Finalize current episode
episode = store.finalize_current_episode(summary="User onboarding conversation")
```

### Emotional Salience

Emotional memories are retained better:

```python
from agent_memory_toolkit.temporal import EmotionalValence

# Add emotionally salient memory
store.add_item(
    content="User successfully completed first task!",
    importance=0.9,
    emotional_valence=EmotionalValence.VERY_POSITIVE,
)

# Get most salient memories
salient = store.get_salient_memories(limit=5)
for item, salience_score in salient:
    print(f"Salience {salience_score:.2f}: {item.content}")
```

## Working Memory

Working memory implements limited capacity with attention-based prioritization.

### Basic Usage

```python
from agent_memory_toolkit.temporal import (
    WorkingMemory,
    WorkingMemoryConfig,
    DisplacementStrategy,
)

# Configure (default: 7 items, Miller's magic number)
config = WorkingMemoryConfig(
    capacity=7,
    auto_rehearsal=True,
    displacement_strategy=DisplacementStrategy.COMBINED,
)

wm = WorkingMemory(config)

# Add items
item1 = wm.add("User wants to book a flight", priority=0.9)
item2 = wm.add("Departure city: New York", priority=0.7, context_tags=["booking"])
item3 = wm.add("Destination: London", priority=0.7, context_tags=["booking"])
```

### Attention and Focus

```python
# Focus attention on specific items
wm.focus(item1.item_id)

# Focused items get activation boost and decay slower
active = wm.get_active(min_activation=0.5)

# Remove focus
wm.unfocus()
```

### Capacity and Displacement

```python
# When capacity is exceeded, items are displaced
wm = WorkingMemory(WorkingMemoryConfig(capacity=3))

wm.add("Item 1", priority=0.5)
wm.add("Item 2", priority=0.8)
wm.add("Item 3", priority=0.3)
wm.add("Item 4", priority=0.9)  # Displaces lowest priority item

# Handle displaced items
def on_displaced(result):
    for item in result.displaced_items:
        print(f"Displaced: {item.content}")
        # Save to long-term memory...

wm.on_displacement(on_displaced)
```

### Decay and Rehearsal

```python
# Apply decay based on elapsed time
low_activation = wm.decay()

# Manual rehearsal strengthens specific items
wm.rehearse(item1.item_id)

# Auto-rehearsal keeps important items active
config = WorkingMemoryConfig(
    auto_rehearsal=True,
    max_rehearsal_items=3,
    rehearsal_interval=timedelta(seconds=30),
)
```

### Context Management

```python
# Set context for current task
context = wm.set_context(
    name="Flight Booking",
    description="Helping user book a flight",
    goals=["Get departure info", "Get destination", "Find flights"],
)

# Filter by context tags
booking_items = wm.get_by_context("booking")

# Snapshot current state
snapshot = wm.snapshot()
print(f"Capacity: {snapshot['capacity_used']}/{snapshot['capacity']}")
```

## Memory Consolidation

The consolidation system transfers memories from working/episodic memory to long-term storage.

### Basic Usage

```python
from agent_memory_toolkit.temporal import (
    MemoryConsolidator,
    ConsolidationConfig,
    ConsolidationStrategy,
    WorkingMemory,
    EpisodicMemoryStore,
    InMemoryLongTermStore,
)

# Set up memory systems
wm = WorkingMemory()
episodes = EpisodicMemoryStore()
ltm = InMemoryLongTermStore()

# Create consolidator
consolidator = MemoryConsolidator(
    working_memory=wm,
    episodic_store=episodes,
    long_term_store=ltm,
)

# Add some memories
wm.add("Important user preference", priority=0.9)
episodes.add_item("Key fact learned", importance=0.8)

# Run consolidation
result = consolidator.consolidate()
print(f"Consolidated {result.consolidated_count} memories")
```

### Consolidation Strategies

```python
from agent_memory_toolkit.temporal import ConsolidationStrategy

config = ConsolidationConfig(
    # Selection strategy
    strategy=ConsolidationStrategy.COMBINED,
    
    # Thresholds
    min_importance=0.3,
    min_activation=0.2,
    
    # Weights for COMBINED strategy
    importance_weight=0.4,
    activation_weight=0.3,
    access_weight=0.2,
    emotional_weight=0.1,
)

consolidator = MemoryConsolidator(config=config)
```

### Automatic Consolidation

```python
# Consolidation triggers automatically on displacement
wm = WorkingMemory(WorkingMemoryConfig(capacity=5))
consolidator = MemoryConsolidator(working_memory=wm)

# Adding items beyond capacity triggers consolidation
for i in range(10):
    wm.add(f"Item {i}", priority=0.5)
# Items are automatically consolidated when displaced
```

### Sleep-Like Consolidation

```python
# Run enhanced consolidation (like sleep memory consolidation)
result = consolidator.run_sleep_consolidation()

# Memories get additional strengthening
print(f"Sleep consolidation strengthened {result.consolidated_count} memories")
```

### Transformation During Consolidation

```python
# Transform/summarize content during consolidation
def summarize(content: str) -> str:
    # Could use LLM for real summarization
    return content[:100] + "..." if len(content) > 100 else content

consolidator.set_transform_callback(summarize)
```

## Integration Example

Combining all temporal memory components:

```python
from datetime import datetime, timedelta
from agent_memory_toolkit.temporal import (
    Timeline,
    TimeWindow,
    TemporalMemory,
    EpisodicMemoryStore,
    EmotionalValence,
    WorkingMemory,
    MemoryConsolidator,
    InMemoryLongTermStore,
)

# Initialize all systems
timeline = Timeline()
episodes = EpisodicMemoryStore()
wm = WorkingMemory()
ltm = InMemoryLongTermStore()

# Connect consolidator
consolidator = MemoryConsolidator(
    working_memory=wm,
    episodic_store=episodes,
    long_term_store=ltm,
)

# Process a conversation
messages = [
    "User: How do I reset my password?",
    "Agent: You can reset it from the settings page.",
    "User: Thanks, that worked!",
]

for msg in messages:
    # Add to timeline for chronological access
    timeline.add(TemporalMemory(
        memory_id=f"msg_{hash(msg)}",
        content=msg,
        timestamp=datetime.now(),
    ))
    
    # Add to episodic memory with appropriate weighting
    is_positive = "thanks" in msg.lower() or "worked" in msg.lower()
    episodes.add_item(
        content=msg,
        importance=0.8 if is_positive else 0.5,
        emotional_valence=EmotionalValence.POSITIVE if is_positive else EmotionalValence.NEUTRAL,
    )
    
    # Key information goes to working memory
    if "password" in msg.lower():
        wm.add(
            "Topic: Password reset",
            priority=0.8,
            context_tags=["support", "password"],
        )

# Query recent from timeline
recent = timeline.query(time_window=TimeWindow.LAST_HOUR)
print(f"Timeline has {len(recent.memories)} recent messages")

# Get strong episodic memories
strong = episodes.get_strong_memories(min_strength=0.5)
print(f"Strong memories: {len(strong)}")

# Active working memory
active = wm.get_active(min_activation=0.5)
print(f"Active WM items: {len(active)}")

# Consolidate for long-term storage
result = consolidator.consolidate()
print(f"Consolidated {result.consolidated_count} to LTM")
```

## Performance Considerations

### 1. Timeline Index Efficiency

The Timeline uses sorted lists with binary search for O(log n) range queries:

```python
# Efficient for large timelines
timeline = Timeline(TimelineConfig(
    index_by_day=True,   # O(1) daily lookups
    index_by_week=True,  # O(1) weekly lookups
    index_by_month=True, # O(1) monthly lookups
))
```

### 2. Batch Operations

```python
# Add many memories at once
memories = [TemporalMemory(...) for _ in range(1000)]
timeline.add_batch(memories)  # More efficient than individual adds
```

### 3. Decay Frequency

```python
# Don't apply decay too frequently
config = EpisodicMemoryConfig(
    # Only decay when explicitly called, or:
    consolidation_interval=timedelta(minutes=5),  # Not too often
)

# Batch decay at intervals rather than per-access
if should_run_decay():
    store.apply_decay()
```

### 4. Working Memory Size

```python
# Larger capacity = more memory, less displacement overhead
config = WorkingMemoryConfig(
    capacity=7,       # Classic Miller's number
    soft_capacity=9,  # Allow slight overflow
)
```

## Best Practices

1. **Use appropriate decay models** - EBBINGHAUS for general use, POWER_LAW for slower forgetting
2. **Set importance ratings** - Higher importance = better retention
3. **Leverage emotional salience** - Emotionally significant memories persist longer
4. **Regular consolidation** - Don't let working memory overflow
5. **Context tagging** - Enables efficient retrieval by topic
6. **Sleep consolidation** - Run periodically for memory strengthening

## See Also

- [Search Guide](search.md) - Combining temporal with semantic search
- [Architecture](architecture.md) - How temporal memory fits in the system
- [API Reference](api-reference.md) - Full API documentation
