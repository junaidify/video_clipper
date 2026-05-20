"""
Video Library Module
Persistent storage for uploaded/downloaded videos.
Each video gets its own directory: {sanitized_title}_{YYYYMMDD_HHMMSS}
Videos can be reused across multiple processing sessions.
"""
import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LIBRARY_INDEX = "library_index.json"


@dataclass
class VideoEntry:
    """A single video in the library."""
    video_id: str           # unique ID (directory name)
    title: str              # original title
    file_path: str          # path to the video file
    directory: str          # video's own directory
    source: str             # 'upload' or 'url'
    source_url: Optional[str] = None
    duration: Optional[float] = None
    added_at: str = ""      # ISO timestamp
    file_size_mb: float = 0.0
    clips_directories: list = field(default_factory=list)  # list of output dirs

    def to_dict(self) -> dict:
        return asdict(self)


class VideoLibrary:
    """Manages persistent video storage and retrieval."""

    def __init__(self, library_root: str):
        self.root = Path(library_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / LIBRARY_INDEX
        self._lock = threading.Lock()
        self._index = self._load_index()

    def _load_index(self) -> dict:
        """Load library index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted library index, starting fresh")
        return {"videos": {}}

    def _save_index(self):
        """Persist library index to disk (caller must hold self._lock)."""
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _sanitize_name(title: str) -> str:
        """Create filesystem-safe name from title."""
        # Remove non-alphanumeric chars (keep spaces and hyphens)
        safe = re.sub(r'[^\w\s-]', '', title)
        # Collapse whitespace
        safe = re.sub(r'\s+', '_', safe.strip())
        # Limit length
        safe = safe[:60] if len(safe) > 60 else safe
        # Fallback if empty
        return safe or "video"

    @staticmethod
    def _generate_video_id(title: str) -> str:
        """Generate unique directory name: {title}_{datetime}."""
        safe_title = VideoLibrary._sanitize_name(title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_title}_{timestamp}"

    def add_video(self, source_path: str, title: str,
                  source: str = "upload",
                  source_url: Optional[str] = None,
                  duration: Optional[float] = None) -> VideoEntry:
        """
        Add a video to the library.
        Copies/moves the file into its own named directory.

        Args:
            source_path: Current path to the video file
            title: Video title (used for directory naming)
            source: 'upload' or 'url'
            source_url: Original URL if downloaded
            duration: Video duration in seconds

        Returns:
            VideoEntry for the stored video
        """
        video_id = self._generate_video_id(title)
        video_dir = self.root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        # Copy video into its directory
        src = Path(source_path)
        dest = video_dir / src.name
        shutil.copy2(str(src), str(dest))

        # Get file size
        file_size_mb = dest.stat().st_size / (1024 * 1024)

        entry = VideoEntry(
            video_id=video_id,
            title=title,
            file_path=str(dest),
            directory=str(video_dir),
            source=source,
            source_url=source_url,
            duration=duration,
            added_at=datetime.now().isoformat(),
            file_size_mb=round(file_size_mb, 2),
            clips_directories=[],
        )

        # Save to index (thread-safe)
        with self._lock:
            self._index["videos"][video_id] = entry.to_dict()
            self._save_index()

        logger.info(f"Video added to library: {video_id} ({file_size_mb:.1f} MB)")
        return entry

    def get_video(self, video_id: str) -> Optional[VideoEntry]:
        """Retrieve a video entry by ID."""
        data = self._index["videos"].get(video_id)
        if not data:
            return None
        return VideoEntry(**data)

    def list_videos(self) -> list:
        """List all videos in the library, newest first."""
        entries = []
        for vid_data in self._index["videos"].values():
            entry = VideoEntry(**vid_data)
            # Verify file still exists
            if Path(entry.file_path).exists():
                entries.append(entry)

        entries.sort(key=lambda e: e.added_at, reverse=True)
        return entries

    def add_clips_directory(self, video_id: str, clips_dir: str):
        """Record a clips output directory for a video."""
        with self._lock:
            if video_id in self._index["videos"]:
                if clips_dir not in self._index["videos"][video_id]["clips_directories"]:
                    self._index["videos"][video_id]["clips_directories"].append(clips_dir)
                    self._save_index()

    def get_clips_for_video(self, video_id: str) -> list:
        """Get all clips directories for a video."""
        data = self._index["videos"].get(video_id, {})
        dirs = data.get("clips_directories", [])
        # Return only directories that still exist
        return [d for d in dirs if Path(d).exists()]

    def delete_video(self, video_id: str) -> bool:
        """Remove a video from the library (deletes files)."""
        with self._lock:
            data = self._index["videos"].get(video_id)
            if not data:
                return False

            video_dir = Path(data["directory"])
            if video_dir.exists():
                shutil.rmtree(str(video_dir), ignore_errors=True)

            del self._index["videos"][video_id]
            self._save_index()
            logger.info(f"Video removed from library: {video_id}")
            return True

    def get_library_stats(self) -> dict:
        """Get library statistics."""
        videos = self.list_videos()
        total_size = sum(v.file_size_mb for v in videos)
        total_clips = sum(len(v.clips_directories) for v in videos)
        return {
            "total_videos": len(videos),
            "total_size_mb": round(total_size, 2),
            "total_clip_sessions": total_clips,
        }
