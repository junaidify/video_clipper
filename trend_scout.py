"""
Trend Scout Module
Pulls trending topics from YouTube, Google Trends, and Reddit.
Deduplicates, scores by momentum, returns ranked topic list.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class TrendingTopic:
    """A single trending topic with metadata."""
    title: str
    source: str                     # 'youtube', 'google_trends', 'reddit'
    score: float = 0.0              # Momentum score (0-100)
    category: str = "entertainment"
    description: str = ""
    url: str = ""
    view_count: int = 0
    engagement: int = 0             # likes, upvotes, etc.
    keywords: list = field(default_factory=list)
    thumbnail: str = ""
    published_at: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "score": round(self.score, 1),
            "category": self.category,
            "description": self.description,
            "url": self.url,
            "view_count": self.view_count,
            "engagement": self.engagement,
            "keywords": self.keywords,
            "thumbnail": self.thumbnail,
            "published_at": self.published_at,
        }


# ── YouTube Trending ──

def fetch_youtube_trending(api_key: str,
                           region: str = "US",
                           category_id: str = "24",
                           max_results: int = 25) -> list[TrendingTopic]:
    """
    Fetch trending videos from YouTube Data API v3.
    Category 24 = Entertainment.
    Other useful categories: 10=Music, 17=Sports, 20=Gaming, 22=People&Blogs
    """
    if not api_key:
        logger.warning("No YouTube API key — skipping YouTube trends")
        return []

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": region,
        "videoCategoryId": category_id,
        "maxResults": min(max_results, 50),
        "key": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        topics = []
        for item in data.get("items", []):
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            # Extract keywords from title
            title = snippet.get("title", "")
            keywords = _extract_keywords(title)

            topic = TrendingTopic(
                title=title,
                source="youtube",
                category=snippet.get("categoryId", "24"),
                description=snippet.get("description", "")[:300],
                url=f"https://youtube.com/watch?v={item['id']}",
                view_count=views,
                engagement=likes + comments,
                keywords=keywords,
                thumbnail=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                published_at=snippet.get("publishedAt", ""),
            )
            # Score: views weighted heavily + engagement bonus
            topic.score = _score_youtube(views, likes, comments)
            topics.append(topic)

        logger.info(f"YouTube trending: {len(topics)} topics fetched")
        return topics

    except Exception as e:
        logger.error(f"YouTube trending fetch failed: {e}")
        return []


def _score_youtube(views: int, likes: int, comments: int) -> float:
    """Score a YouTube video by virality potential. Max ~100."""
    import math
    view_score = min(40, math.log10(max(views, 1)) * 5)
    like_score = min(30, math.log10(max(likes, 1)) * 5)
    comment_score = min(20, math.log10(max(comments, 1)) * 6)
    # Engagement ratio bonus
    ratio = (likes + comments) / max(views, 1)
    ratio_bonus = min(10, ratio * 500)
    return min(100, view_score + like_score + comment_score + ratio_bonus)


# ── Google Trends ──

def fetch_google_trends(region: str = "united_states",
                        category: str = "entertainment",
                        max_results: int = 20) -> list[TrendingTopic]:
    """
    Fetch trending searches from Google Trends via unofficial RSS/JSON endpoint.
    Falls back to pytrends if available.
    """
    topics = []

    # Try Google Trends daily trends RSS (no API key needed)
    try:
        geo = _region_to_geo(region)
        rss_url = f"https://trends.google.com/trending/rss?geo={geo}"
        resp = requests.get(rss_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        if resp.status_code == 200:
            topics = _parse_trends_rss(resp.text, max_results)
            logger.info(f"Google Trends RSS: {len(topics)} topics")
    except Exception as e:
        logger.warning(f"Google Trends RSS failed: {e}")

    # Fallback: try pytrends library
    if not topics:
        try:
            topics = _fetch_pytrends(region, max_results)
        except Exception as e:
            logger.warning(f"pytrends fallback failed: {e}")

    return topics


def _parse_trends_rss(xml_text: str, max_results: int) -> list[TrendingTopic]:
    """Parse Google Trends RSS XML into TrendingTopic list."""
    import xml.etree.ElementTree as ET
    topics = []

    try:
        root = ET.fromstring(xml_text)
        ns = {"ht": "https://trends.google.com/trending/rss"}

        for item in root.findall(".//item")[:max_results]:
            title = item.findtext("title", "").strip()
            if not title:
                continue

            traffic = item.findtext(f"{{{ns['ht']}}}approx_traffic", "0")
            traffic_num = int(re.sub(r"[^\d]", "", traffic) or "0")

            topic = TrendingTopic(
                title=title,
                source="google_trends",
                category="entertainment",
                description=item.findtext("description", "")[:300],
                url=item.findtext("link", ""),
                view_count=traffic_num,
                keywords=_extract_keywords(title),
            )
            topic.score = min(100, (traffic_num / 10000) * 10 + 40)
            topics.append(topic)
    except ET.ParseError as e:
        logger.warning(f"Trends RSS parse error: {e}")

    return topics


def _fetch_pytrends(region: str, max_results: int) -> list[TrendingTopic]:
    """Fetch using pytrends library if installed."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        # pytrends expects lowercase country names like 'united_states'
        geo = _region_to_geo(region)
        geo_to_pytrends = {
            "US": "united_states", "GB": "united_kingdom", "IN": "india",
            "CA": "canada", "AU": "australia", "DE": "germany",
            "FR": "france", "JP": "japan", "BR": "brazil",
        }
        pn = geo_to_pytrends.get(geo.upper(), "united_states")
        df = pytrends.trending_searches(pn=pn)

        topics = []
        for i, row in df.head(max_results).iterrows():
            title = str(row[0]).strip()
            topic = TrendingTopic(
                title=title,
                source="google_trends",
                category="entertainment",
                keywords=_extract_keywords(title),
            )
            # Position-based score (top = higher)
            topic.score = max(30, 80 - (i * 3))
            topics.append(topic)

        logger.info(f"pytrends: {len(topics)} trending searches")
        return topics
    except ImportError:
        logger.info("pytrends not installed — skipping")
        return []


# ── Reddit Trending ──

def fetch_reddit_trending(subreddits: list[str] = None,
                          region: str = "US",
                          max_results: int = 20,
                          time_filter: str = "day") -> list[TrendingTopic]:
    """
    Fetch hot posts from entertainment subreddits.
    Uses Reddit's public JSON API (no auth needed for .json endpoints).
    Region-aware: uses localized subreddits when available.
    """
    if subreddits is None:
        # Region-specific subreddit mapping
        region_subs = {
            "US": ["entertainment", "videos", "movies", "television", "popculture", "celebrity", "Music"],
            "GB": ["entertainment", "CasualUK", "BritishTV", "UKFilm", "unitedkingdom", "videos", "Music"],
            "IN": ["india", "bollywood", "IndianEntertainment", "BollywoodMusic", "videos", "Music"],
            "CA": ["canada", "entertainment", "videos", "movies", "Music", "television"],
            "AU": ["australia", "entertainment", "AustralianTV", "videos", "Music", "movies"],
            "DE": ["de", "entertainment", "german", "videos", "Music", "movies"],
            "FR": ["france", "entertainment", "videos", "Music", "movies", "television"],
            "JP": ["japan", "entertainment", "videos", "Music", "movies", "anime"],
            "BR": ["brasil", "entertainment", "videos", "Music", "movies", "television"],
        }
        geo = region.upper()[:2]
        subreddits = region_subs.get(geo, region_subs["US"])

    topics = []
    per_sub = max(3, max_results // len(subreddits))

    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub}&t={time_filter}"
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "VideoClipper/1.0 (trending research)"
            })

            if resp.status_code != 200:
                continue

            data = resp.json()
            for post in data.get("data", {}).get("children", []):
                pdata = post.get("data", {})
                if pdata.get("stickied"):
                    continue

                title = pdata.get("title", "").strip()
                if not title:
                    continue

                ups = pdata.get("ups", 0)
                comments = pdata.get("num_comments", 0)

                topic = TrendingTopic(
                    title=title,
                    source="reddit",
                    category="entertainment",
                    description=pdata.get("selftext", "")[:300],
                    url=f"https://reddit.com{pdata.get('permalink', '')}",
                    engagement=ups + comments,
                    keywords=_extract_keywords(title),
                    thumbnail=pdata.get("thumbnail", "")
                        if pdata.get("thumbnail", "").startswith("http") else "",
                    published_at=str(pdata.get("created_utc", "")),
                )
                topic.score = _score_reddit(ups, comments)
                topics.append(topic)

            time.sleep(0.5)  # Rate limit courtesy

        except Exception as e:
            logger.warning(f"Reddit r/{sub} fetch failed: {e}")

    logger.info(f"Reddit trending: {len(topics)} topics from {len(subreddits)} subs")
    return topics


def _score_reddit(upvotes: int, comments: int) -> float:
    """Score Reddit post by virality. Max ~100."""
    import math
    up_score = min(50, math.log10(max(upvotes, 1)) * 12)
    comment_score = min(30, math.log10(max(comments, 1)) * 10)
    # High comment-to-upvote ratio = controversial/engaging
    ratio = comments / max(upvotes, 1)
    ratio_bonus = min(20, ratio * 50)
    return min(100, up_score + comment_score + ratio_bonus)


# ── Aggregation & Deduplication ──

def scout_trending(youtube_api_key: str = None,
                   region: str = "US",
                   max_per_source: int = 15,
                   min_score: float = 20.0) -> list[dict]:
    """
    Master function: fetch from all sources, deduplicate, rank.

    Returns sorted list of topic dicts, highest score first.
    """
    if youtube_api_key is None:
        youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")

    all_topics = []

    # Normalize region to 2-letter ISO code for consistency
    geo = _region_to_geo(region)

    # Fetch from all three sources — pass normalized geo to each
    yt = fetch_youtube_trending(youtube_api_key, geo, max_results=max_per_source)
    gt = fetch_google_trends(region=geo, max_results=max_per_source)
    rd = fetch_reddit_trending(region=geo, max_results=max_per_source)

    all_topics.extend(yt)
    all_topics.extend(gt)
    all_topics.extend(rd)

    if not all_topics:
        logger.warning("No trending topics found from any source")
        return []

    # Deduplicate by fuzzy title matching
    deduped = _deduplicate(all_topics)

    # Filter by minimum score
    filtered = [t for t in deduped if t.score >= min_score]

    # Sort by score descending
    filtered.sort(key=lambda t: t.score, reverse=True)

    logger.info(
        f"Trend Scout: {len(all_topics)} raw → {len(deduped)} deduped → "
        f"{len(filtered)} above threshold"
    )

    return [t.to_dict() for t in filtered]


def _deduplicate(topics: list[TrendingTopic]) -> list[TrendingTopic]:
    """
    Merge duplicate topics across sources.
    If same topic appears on YouTube AND Reddit, boost its score.
    """
    seen = {}  # normalized_title -> TrendingTopic

    for topic in topics:
        key = _normalize_title(topic.title)

        if key in seen:
            existing = seen[key]
            # Cross-source bonus: +15 for appearing on multiple platforms
            if existing.source != topic.source:
                existing.score = min(100, existing.score + 15)
            # Keep higher engagement numbers
            existing.view_count = max(existing.view_count, topic.view_count)
            existing.engagement = max(existing.engagement, topic.engagement)
            # Merge keywords
            existing.keywords = list(set(existing.keywords + topic.keywords))
            # Keep the better thumbnail
            if not existing.thumbnail and topic.thumbnail:
                existing.thumbnail = topic.thumbnail
        else:
            seen[key] = topic

    return list(seen.values())


def _normalize_title(title: str) -> str:
    """Normalize title for dedup matching."""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    # Remove common filler words
    for word in ["the", "a", "an", "is", "are", "was", "in", "on", "at", "to", "for"]:
        t = re.sub(rf"\b{word}\b", "", t)
    return t.strip()


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a title/text."""
    # Remove special chars, split, filter short words
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).split()
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "this", "that", "with",
        "from", "by", "be", "as", "it", "its", "has", "have", "had",
        "not", "no", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "just", "so", "than", "then",
        "new", "now", "how", "what", "why", "who", "when", "where",
        "all", "every", "each", "most", "some", "any", "about", "out",
        "up", "if", "my", "your", "his", "her", "our", "they", "we",
        "you", "me", "him", "them", "us", "been", "being", "get", "got",
    }
    keywords = [w.lower() for w in words if len(w) > 2 and w.lower() not in stop_words]
    return list(dict.fromkeys(keywords))[:10]  # Unique, max 10


def _region_to_geo(region: str) -> str:
    """Convert region string to geo code."""
    mapping = {
        "united_states": "US", "us": "US",
        "united_kingdom": "GB", "uk": "GB", "gb": "GB",
        "india": "IN", "in": "IN",
        "canada": "CA", "ca": "CA",
        "australia": "AU", "au": "AU",
        "germany": "DE", "de": "DE",
        "france": "FR", "fr": "FR",
        "japan": "JP", "jp": "JP",
        "brazil": "BR", "br": "BR",
    }
    return mapping.get(region.lower(), region.upper()[:2])


def get_available_categories() -> list[dict]:
    """Return available YouTube video categories for UI dropdown."""
    return [
        {"id": "24", "name": "Entertainment"},
        {"id": "10", "name": "Music"},
        {"id": "17", "name": "Sports"},
        {"id": "20", "name": "Gaming"},
        {"id": "22", "name": "People & Blogs"},
        {"id": "23", "name": "Comedy"},
        {"id": "25", "name": "News & Politics"},
        {"id": "28", "name": "Science & Technology"},
        {"id": "1", "name": "Film & Animation"},
        {"id": "26", "name": "Howto & Style"},
    ]
