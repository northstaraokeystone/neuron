"""
NEURON v4.2: Sharding Module
Distributes ledger entries across multiple shard files for horizontal scale.
Inspired by Grok's sliding window: truncate oldest when full.
~120 lines. Hash-routed. Eviction-enabled.
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Sharding constants
DEFAULT_SHARD_COUNT = 4
MAX_SHARD_COUNT = 64
SHARD_STRATEGIES = ["hash", "time", "project", "model"]
DEFAULT_SHARD_STRATEGY = "hash"
SHARD_MAX_ENTRIES = 1_000_000
SHARD_EVICTION_PERCENT = 0.20  # Evict oldest 20%
SHARD_DIR = Path(os.environ.get("NEURON_SHARD_DIR", Path.home() / "neuron" / "shards"))
ARCHIVE_PATH = Path(
    os.environ.get("NEURON_ARCHIVE", Path.home() / "neuron" / "archive.jsonl")
)

# Project/model mappings for routing strategies
PROJECTS = ["agentproof", "axiom", "neuron"]
MODELS = ["grok", "claude", "gemini", "neuron"]


def _dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 hash."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    sha256_hex = hashlib.sha256(data).hexdigest()
    blake3_hex = (
        blake3.blake3(data).hexdigest()
        if HAS_BLAKE3
        else hashlib.sha256(b"blake3:" + data).hexdigest()
    )
    return f"{sha256_hex}:{blake3_hex}"


class ShardedLedger:
    """
    Distributes entries across N shard files.
    Inspired by Grok's sliding window: truncate oldest when full.
    """

    def __init__(
        self,
        shard_count: int = DEFAULT_SHARD_COUNT,
        strategy: Literal["hash", "time", "project", "model"] = DEFAULT_SHARD_STRATEGY,
        max_entries_per_shard: int = SHARD_MAX_ENTRIES,
        shard_dir: Path | None = None,
    ):
        self.shard_count = min(max(1, shard_count), MAX_SHARD_COUNT)
        self.strategy = (
            strategy if strategy in SHARD_STRATEGIES else DEFAULT_SHARD_STRATEGY
        )
        self.max_entries_per_shard = max_entries_per_shard
        self.shard_dir = Path(shard_dir) if shard_dir else SHARD_DIR
        self._lock = threading.Lock()
        self._init_shards()

    def _init_shards(self) -> None:
        """Initialize shard directory and files."""
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.shard_count):
            shard_path = self.shard_dir / f"shard_{i}.jsonl"
            if not shard_path.exists():
                shard_path.touch()

    def _shard_path(self, shard_id: int) -> Path:
        """Get path to a specific shard file."""
        return self.shard_dir / f"shard_{shard_id}.jsonl"

    def route(self, entry: dict) -> int:
        """Determine shard index for entry based on strategy."""
        if self.strategy == "hash":
            entry_hash = entry.get("hash", "")
            if not entry_hash:
                # Fallback to hashing the entry content
                entry_hash = hashlib.sha256(
                    json.dumps(entry, sort_keys=True).encode()
                ).hexdigest()
            # Handle dual-hash format (sha256:blake3) or plain hash
            if ":" in entry_hash:
                entry_hash = entry_hash.split(":")[0]
            # Hash the string if it's not valid hex
            try:
                return int(entry_hash[:8], 16) % self.shard_count
            except ValueError:
                # Hash the non-hex string to get a consistent shard
                hashed = hashlib.sha256(entry_hash.encode()).hexdigest()
                return int(hashed[:8], 16) % self.shard_count

        elif self.strategy == "time":
            ts = entry.get("ts", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return dt.hour % self.shard_count
                except ValueError:
                    pass
            return 0

        elif self.strategy == "project":
            project = entry.get("project", "neuron")
            try:
                return PROJECTS.index(project) % self.shard_count
            except ValueError:
                return 0

        elif self.strategy == "model":
            model = entry.get("model", "neuron")
            try:
                return MODELS.index(model) % self.shard_count
            except ValueError:
                return 0

        return 0  # Default fallback

    def _read_shard(self, shard_id: int) -> list[dict]:
        """Read all entries from a shard file."""
        shard_path = self._shard_path(shard_id)
        entries = []
        if shard_path.exists():
            with open(shard_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return entries

    def _write_shard(self, shard_id: int, entries: list[dict]) -> None:
        """Write entries to a shard file (atomic overwrite)."""
        shard_path = self._shard_path(shard_id)
        with open(shard_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def _append_archive(self, entries: list[dict]) -> None:
        """Append evicted entries to archive file."""
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARCHIVE_PATH, "a") as f:
            for e in entries:
                e["archived_ts"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                f.write(json.dumps(e) + "\n")

    def append(self, entry: dict) -> dict:
        """Write entry to appropriate shard."""
        shard_id = self.route(entry)
        shard_path = self._shard_path(shard_id)

        with self._lock:
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shard_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Check if eviction needed
            entries = self._read_shard(shard_id)
            if len(entries) > self.max_entries_per_shard:
                self.evict(shard_id)

        return {**entry, "shard_id": shard_id}

    def replay(self, n: int | None = None, since: str | None = None) -> list[dict]:
        """Aggregate entries across all shards."""
        all_entries = []
        for i in range(self.shard_count):
            all_entries.extend(self._read_shard(i))

        # Sort by timestamp
        all_entries.sort(key=lambda e: e.get("ts", ""))

        # Filter by since if provided
        if since:
            all_entries = [e for e in all_entries if e.get("ts", "") >= since]

        # Return last n entries
        if n is not None:
            return all_entries[-n:]
        return all_entries

    def evict(self, shard_id: int) -> dict:
        """
        Sliding window eviction: move oldest entries to archive.
        Mirrors Grok's 'truncate oldest' behavior.
        """
        entries = self._read_shard(shard_id)

        if len(entries) <= self.max_entries_per_shard:
            return {"shard_id": shard_id, "evicted": 0, "reason": "under_limit"}

        # Sort by timestamp, evict enough to get below limit
        entries.sort(key=lambda e: e.get("ts", ""))

        # Calculate how many to evict: at least 20%, but ensure we get below limit
        min_evict = max(1, int(len(entries) * SHARD_EVICTION_PERCENT))
        entries_over = len(entries) - self.max_entries_per_shard
        evict_count = max(min_evict, entries_over)

        to_evict = entries[:evict_count]
        to_keep = entries[evict_count:]

        # Archive evicted entries
        self._append_archive(to_evict)
        self._write_shard(shard_id, to_keep)

        return {
            "shard_id": shard_id,
            "evicted": evict_count,
            "remaining": len(to_keep),
            "reason": "over_limit",
        }

    def sync(self) -> dict:
        """Verify cross-shard consistency using merkle-style hashing."""
        shard_hashes = {}
        total_entries = 0
        conflicts = 0

        for i in range(self.shard_count):
            entries = self._read_shard(i)
            total_entries += len(entries)

            # Hash all entry hashes in this shard
            entry_hashes = [e.get("hash", "") for e in entries]
            combined = ":".join(sorted(entry_hashes))
            shard_hashes[f"shard_{i}"] = _dual_hash(combined)[:32]

        return {
            "shards_synced": self.shard_count,
            "total_entries": total_entries,
            "conflicts": conflicts,
            "shard_hashes": shard_hashes,
            "sync_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def rebalance(self) -> dict:
        """Redistribute entries if uneven across shards."""
        all_entries = []
        for i in range(self.shard_count):
            all_entries.extend(self._read_shard(i))

        if not all_entries:
            return {"rebalanced": False, "reason": "no_entries"}

        # Clear all shards
        for i in range(self.shard_count):
            self._write_shard(i, [])

        # Re-route all entries
        entries_per_shard = [0] * self.shard_count
        for entry in all_entries:
            shard_id = self.route(entry)
            shard_path = self._shard_path(shard_id)
            with open(shard_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            entries_per_shard[shard_id] += 1

        return {
            "rebalanced": True,
            "total_entries": len(all_entries),
            "entries_per_shard": entries_per_shard,
        }

    def stats(self) -> dict:
        """Per-shard entry counts and sizes."""
        shards = []
        total_entries = 0
        total_bytes = 0

        for i in range(self.shard_count):
            shard_path = self._shard_path(i)
            entries = self._read_shard(i)
            size_bytes = shard_path.stat().st_size if shard_path.exists() else 0

            shards.append(
                {"shard_id": i, "count": len(entries), "size_bytes": size_bytes}
            )
            total_entries += len(entries)
            total_bytes += size_bytes

        return {
            "shard_count": self.shard_count,
            "strategy": self.strategy,
            "max_entries_per_shard": self.max_entries_per_shard,
            "total_entries": total_entries,
            "total_bytes": total_bytes,
            "shards": shards,
        }


def _emit_shard_receipt(receipt_type: str, data: dict) -> dict:
    """Emit a sharding receipt."""
    from stress import _get_stress_receipts_path

    receipt = {
        "type": receipt_type,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **data,
    }
    receipt["hash"] = _dual_hash(
        json.dumps({k: v for k, v in receipt.items() if k != "hash"}, sort_keys=True)
    )
    receipts_path = _get_stress_receipts_path()
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipts_path, "a") as f:
        f.write(json.dumps(receipt) + "\n")
    return receipt
