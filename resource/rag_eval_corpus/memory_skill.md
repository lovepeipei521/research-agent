# Memory and skill layers

Poirot memory has five layers. L1 defines MemoryTrace and atomic operations.
L2 provides decay and forgetting strategies. L3 stores Markdown truth source
and uses a pure BM25 HybridRetriever. L4 injects recalled memories through a
before_model middleware. L5 submits consolidation work to a daemon worker and
threading.Queue so the main agent loop is not blocked.

Skills are research process knowledge bundles, not executable functions. The
Skill system has a base layer, an evolution layer, and an evaluation layer.
Layer 3 evaluates execution judgment, four-dimensional task quality, and
response contracts through RuntimeTracker.
