# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""S3 / MinIO object-store adapter (Phase 5, production backend).

Implements ``pipeline.store.objectstore.ObjectStore`` over an S3-compatible API (AWS S3,
MinIO, Ceph RGW). The boto3 client is **injected**, so the adapter's key-mapping and listing
logic is fully unit-tested with an in-memory fake client — no boto3 and no live bucket needed
in CI. ``from_env`` lazily constructs a real boto3 client for production use.
"""

from __future__ import annotations


class S3ObjectStore:
    """ObjectStore over an S3-compatible ``client`` (boto3-style: put_object/get_object/...).

    Keys are stored under ``prefix`` within ``bucket``. ``client`` must expose
    ``put_object(Bucket, Key, Body)``, ``get_object(Bucket, Key)``, ``head_object(Bucket, Key)``,
    and ``list_objects_v2(Bucket, Prefix)`` — boto3's S3 client and MinIO satisfy this.
    """

    def __init__(self, bucket: str, client, *, prefix: str = ""):
        self.bucket = bucket
        self.client = client
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        body = resp["Body"]
        return body.read() if hasattr(body, "read") else body

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[str]:
        full = self._key(prefix)
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full)
        out = []
        for obj in resp.get("Contents", []) or []:
            k = obj["Key"]
            out.append(k[len(self.prefix):] if self.prefix and k.startswith(self.prefix) else k)
        return sorted(out)

    # ---- shard conveniences (mirror LocalObjectStore) ---- #

    def put_shard(self, key: str, docs) -> int:
        import json

        lines = [json.dumps(d, ensure_ascii=False, sort_keys=True) for d in docs]
        self.put(key, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        return len(lines)

    def get_shard(self, key: str) -> list[dict]:
        import json

        return [json.loads(ln) for ln in self.get(key).decode("utf-8").splitlines() if ln.strip()]

    @classmethod
    def from_env(cls, bucket: str, *, prefix: str = "", endpoint_url: str | None = None):
        """Construct with a real boto3 S3 client (lazily imported). For production use."""
        try:
            import boto3
        except Exception as e:  # pragma: no cover - exercised only with boto3 installed
            raise RuntimeError("S3ObjectStore.from_env requires boto3") from e
        client = boto3.client("s3", endpoint_url=endpoint_url)
        return cls(bucket, client, prefix=prefix)


__all__ = ["S3ObjectStore"]
