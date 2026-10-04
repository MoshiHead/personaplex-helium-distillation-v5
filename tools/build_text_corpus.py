#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the text corpus for the student's text-only distillation stage.

    python tools/build_text_corpus.py                       # -> /workspace/ppx_text_corpus (or ./ppx_text_corpus)
    python tools/build_text_corpus.py --out /some/dir
    python tools/build_text_corpus.py --no-fetch            # rebuild from the raw cache, no network

Why a dedicated corpus
----------------------
The student's temporal transformer is the only part of PersonaPlex that holds world knowledge, and
`distill/init_from_teacher.py` reduces it to ~10% of the teacher's parameters (FFN memory slots: 20.8%).
Recovering its text behaviour needs text tokens, and the 100 h audio corpus holds well under 1M agent text
tokens in total -- ~90% of its text stream is PAD/EPAD. Text-only steps (`distill/train.py --text-data`) feed
the same transformer one position per token with no Mimi and no depformer, so a real text corpus fits in the
same GPU budget.

The objective is KL against the teacher, so this corpus does NOT need answers in it -- it only has to COVER
the domain. What matters is that it is explanatory prose about the right subjects.

Sources (all free-culture licensed, all human-written explanatory prose)
-----------------------------------------------------------------------
1. English Wikipedia, via the MediaWiki `extracts` API (CC BY-SA 4.0) -- breadth across every requested
   topic, plus history, Satoshi Nakamoto, Ethereum, stablecoins and security.
2. "Mastering Bitcoin", 3rd edition, Antonopoulos/Harding (CC BY-SA 4.0), AsciiDoc from the official repo --
   Bitcoin depth: keys, wallets, transactions, signatures, the network, the blockchain, mining and proof of
   work, security. Appendix A is the Bitcoin whitepaper itself.
3. ethereum.org developer documentation (CC BY 4.0), Markdown from the official repo -- Ethereum depth:
   accounts, transactions, gas, smart contracts, the EVM, proof of stake, standards, scaling.

Deliberately NOT used: price/market data, OHLC series, charts, tweets/Reddit/forum scrapes, SEO spam,
"top 10 coins" listicles, and noisy scraped crypto datasets. They teach the model nothing about what Bitcoin
IS, and price tables in particular are mostly digits -- which is why `_quality_ok` rejects digit-heavy text.

Output (exactly what distill/data/text_dataset.py reads)
-------------------------------------------------------
`<out>/*.jsonl`, one JSON object per line with a `"text"` key -- the first key `_iter_documents` looks for.
One line = one document, and `build_token_cache` appends SEPARATOR (text id 3) after each, so document
boundaries land on the token the model already reads as "nothing being said". Extra keys (`title`, `topic`,
`source`, `license`) are ignored by the loader and kept for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import typing as tp
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "PersonaPlexDistillCorpus/1.0 (student-distillation research corpus builder)"

# ----------------------------------------------------------------------------------------- source definitions

# topic -> Wikipedia article titles. Topics are the ones the corpus must cover; titles are resolved through
# redirects by the API (`redirects=1`), so e.g. "Bitcoin mining" lands on the right article.
WIKIPEDIA: dict[str, list[str]] = {
    "bitcoin": [
        "Bitcoin", "Bitcoin network", "Bitcoin protocol", "Bitcoin Core", "Bitcoin scalability problem",
        "Lightning Network", "SegWit", "Bitcoin ATM", "Legality of cryptocurrency by country or territory",
    ],
    "bitcoin_history": [
        "History of bitcoin", "Satoshi Nakamoto", "Genesis block", "Mt. Gox", "Cypherpunk",
        "Crypto-anarchism", "Hashcash", "B-money", "DigiCash",
        # "Bit Gold" is not an article; its inventor's page is where that history lives.
        "Nick Szabo",
    ],
    "bitcoin_transactions": [
        "Unspent transaction output", "Double-spending", "Merkle tree", "Transaction fee",
    ],
    "blockchain": [
        "Blockchain", "Distributed ledger", "Fork (blockchain)", "Block (data storage)",
        "Consensus (computer science)", "Byzantine fault", "Peer-to-peer", "Distributed computing",
    ],
    "crypto_fundamentals": [
        "Cryptocurrency", "Digital currency", "Virtual currency", "Cryptocurrency exchange",
        "Initial coin offering", "Central bank digital currency",
        "Decentralized application", "Decentralized autonomous organization", "Cryptocurrency bubble",
        "Litecoin", "Monero", "Dogecoin", "Ripple Labs", "Cardano (blockchain platform)",
        "Solana (blockchain platform)", "Binance", "Coinbase",
        # "Token (blockchain)" is not an article; these cover token platforms concretely instead.
        "Tron (blockchain)", "Polygon (blockchain)",
    ],
    "wallets_and_keys": [
        # Seed phrases / BIP-32 have no standalone article; "Cryptocurrency wallet" plus Mastering
        # Bitcoin ch04-ch05 cover deterministic wallets in depth.
        "Cryptocurrency wallet", "Public-key cryptography", "Digital signature",
        "Elliptic-curve cryptography", "Elliptic Curve Digital Signature Algorithm", "Key (cryptography)",
        "Cryptographic hash function", "SHA-2", "Base58", "Key derivation function", "Cryptography",
    ],
    "mining_and_pow": [
        "Proof of work", "Mining pool", "Application-specific integrated circuit",
        "Environmental effects of bitcoin", "Cryptocurrency and crime", "Proof of stake",
        "Proof of space",
    ],
    "ethereum": [
        "Ethereum", "Smart contract", "Solidity", "ERC-20", "Non-fungible token",
        "Decentralized finance", "Ethereum Classic", "The DAO (organization)",
    ],
    "stablecoins": [
        "Stablecoin", "Tether (cryptocurrency)", "USD Coin", "Dai (cryptocurrency)", "TerraUSD",
    ],
    "crypto_security": [
        "Phishing", "Social engineering (security)", "Ponzi scheme", "Confidence trick",
        "Money laundering", "Know your customer", "Multi-factor authentication", "Cold boot attack",
        "Hardware security module", "Pig butchering scam",
    ],
    # ---- finance / economics: the vocabulary a crypto educator keeps reaching for -------------------
    "finance_economics": [
        "Money", "Currency", "Fiat money", "Inflation", "Deflation", "Monetary policy", "Central bank",
        "Federal Reserve", "Interest rate", "Bank", "Commercial bank", "Payment system", "Credit card",
        "Wire transfer", "Remittance", "Foreign exchange market", "Stock", "Stock market", "Bond (finance)",
        "Exchange-traded fund", "Asset", "Volatility (finance)", "Market capitalization", "Liquidity",
        "Supply and demand", "Store of value", "Medium of exchange", "Gold as an investment",
        "Hyperinflation", "Financial regulation", "Securities regulation in the United States",
        "Commodity Futures Trading Commission", "U.S. Securities and Exchange Commission", "Tax",
        "Capital gains tax", "Double-entry bookkeeping", "Ledger", "Audit", "Escrow", "Derivative (finance)",
        "Futures contract", "Short (finance)", "Leverage (finance)", "Arbitrage", "Speculation",
        "Diversification (finance)", "Risk management", "Due diligence", "Fraud",
    ],
    # ---- computing / networking / crypto primitives --------------------------------------------------
    "computing": [
        "Computer network", "Internet", "Internet protocol suite", "Transmission Control Protocol",
        "Client-server model", "Server (computing)", "Database", "Transaction processing",
        "ACID", "Replication (computing)", "Fault tolerance", "Scalability", "Latency (engineering)",
        "Throughput", "Encryption", "Symmetric-key algorithm", "RSA (cryptosystem)", "Diffie-Hellman key exchange",
        "Transport Layer Security", "Random number generation", "Cryptographically secure pseudorandom number generator",
        "Hash table", "Data structure", "Algorithm", "Open-source software", "Version control",
        "Application programming interface", "Virtual machine", "Compiler", "Turing completeness",
        "Zero-knowledge proof", "Homomorphic encryption", "Multi-party computation", "Quantum computing",
        "Post-quantum cryptography", "Denial-of-service attack", "Man-in-the-middle attack", "Sybil attack",
        "Computer security", "Authentication", "Password", "Two-man rule",
    ],
    # ---- general knowledge: the student is asked open questions too ("capital city of Japan?"),
    # and a broader text distribution is what keeps the text pathway from collapsing into one domain.
    "general_knowledge": [
        "Japan", "Tokyo", "China", "Beijing", "India", "New Delhi", "United States", "Washington, D.C.",
        "United Kingdom", "London", "France", "Paris", "Germany", "Berlin", "Brazil", "Russia", "Canada",
        "Australia", "Nigeria", "Egypt", "Capital city", "Country", "Continent", "Earth", "Geography",
        "History", "Science", "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy", "Solar System",
        "Sun", "Moon", "Water", "Electricity", "Energy", "Climate change", "Evolution", "DNA", "Human body",
        "Medicine", "Vaccine", "Language", "English language", "Writing", "Printing press", "Industrial Revolution",
        "Electricity generation", "Artificial intelligence", "Machine learning", "Computer",
        "Telephone", "Television", "Automobile", "Airplane", "Railway", "Agriculture", "Food", "Cooking",
        "Music", "Literature", "Philosophy", "Religion", "Democracy", "Government", "Law", "Education",
        "Sport", "Association football", "Olympic Games", "Time", "Calendar", "Measurement",
    ],
}

# "Mastering Bitcoin" 3rd edition -- AsciiDoc sources. appa_whitepaper is the Bitcoin whitepaper.
MASTERING_BITCOIN_RAW = "https://raw.githubusercontent.com/bitcoinbook/bitcoinbook/develop/"
MASTERING_BITCOIN: dict[str, list[str]] = {
    "crypto_fundamentals":     ["ch01_intro.adoc", "ch02_overview.adoc"],
    "bitcoin":                 ["ch03_bitcoin-core.adoc", "ch10_network.adoc"],
    "wallets_and_keys":        ["ch04_keys.adoc", "ch05_wallets.adoc", "ch08_signatures.adoc"],
    "bitcoin_transactions":    ["ch06_transactions.adoc", "ch07_authorization-authentication.adoc",
                                "ch09_fees.adoc"],
    "blockchain":              ["ch11_blockchain.adoc"],
    "mining_and_pow":          ["ch12_mining.adoc"],
    "crypto_security":         ["ch13_security.adoc"],
    "ethereum":                ["ch14_applications.adoc"],
    "bitcoin_history":         ["appa_whitepaper.adoc"],
}

# ethereum.org developer docs -- Markdown sources (path under public/content/, index.md implied).
ETHEREUM_ORG_RAW = "https://raw.githubusercontent.com/ethereum/ethereum-org-website/dev/public/content/"
ETHEREUM_ORG: dict[str, list[str]] = {
    "ethereum": [
        "developers/docs/intro-to-ethereum", "developers/docs/intro-to-ether",
        "developers/docs/accounts", "developers/docs/transactions", "developers/docs/blocks",
        "developers/docs/gas", "developers/docs/evm", "developers/docs/evm/opcodes",
        "developers/docs/smart-contracts", "developers/docs/smart-contracts/anatomy",
        "developers/docs/smart-contracts/languages", "developers/docs/smart-contracts/compiling",
        "developers/docs/smart-contracts/deploying", "developers/docs/smart-contracts/libraries",
        "developers/docs/standards/tokens/erc-20", "developers/docs/standards/tokens/erc-721",
        "developers/docs/standards/tokens/erc-1155",
        "developers/docs/dapps", "developers/docs/web2-vs-web3", "developers/docs/networks",
        "developers/docs/nodes-and-clients", "developers/docs/nodes-and-clients/node-architecture",
        "developers/docs/scaling", "developers/docs/scaling/optimistic-rollups",
        "developers/docs/scaling/zk-rollups",
        "developers/docs/data-availability",
    ],
    "mining_and_pow": [
        "developers/docs/consensus-mechanisms", "developers/docs/consensus-mechanisms/pos",
        "developers/docs/consensus-mechanisms/pow", "developers/docs/consensus-mechanisms/pos/attack-and-defense",
        "developers/docs/consensus-mechanisms/pos/rewards-and-penalties",
        "developers/docs/consensus-mechanisms/pow/mining",
    ],
    "blockchain": ["developers/docs/data-structures-and-encoding/patricia-merkle-trie",
                   "developers/docs/data-structures-and-encoding/rlp", "developers/docs/bridges",
                   "roadmap/merge"],
    "wallets_and_keys": ["developers/docs/apis/json-rpc"],
    # `security` is the top-level user-facing security page; the /wallets/ page is not a raw index.md
    # (verified 404), so wallet material comes from Wikipedia + Mastering Bitcoin instead.
    "crypto_security": ["security", "developers/docs/smart-contracts/security",
                        "developers/docs/smart-contracts/testing", "developers/docs/mev"],
    "crypto_fundamentals": ["developers/docs/standards/tokens", "developers/docs/oracles"],
}

# Bitcoin Improvement Proposals (MediaWiki) and Ethereum Improvement Proposals (Markdown): the normative
# specs behind everything the teacher explains, each written as prose with a motivation and rationale
# section. Listed through the GitHub contents API, then pulled from raw.githubusercontent.
BIPS_REPO = "bitcoin/bips"
EIPS_REPO = "ethereum/EIPs"
# Most EIPs are short stubs or withdrawn drafts; only files above these sizes carry explanatory prose.
EIP_MIN_BYTES = 6000
BIP_MIN_BYTES = 4000

LICENSES = {
    "wikipedia": "CC BY-SA 4.0",
    "bips": "public domain / CC0, per each proposal's Licence header (BIP-2)",
    "eips": "CC0 1.0",
    "mastering_bitcoin": "CC BY-SA 4.0",
    "ethereum_org": "CC BY 4.0",
}

# ------------------------------------------------------------------------------------------------ fetching


def fetch(url: str, retries: int = 6, pause: float = 1.0) -> str | None:
    """GET with a real User-Agent and 429-aware exponential backoff.

    Wikipedia's API rate-limits anonymous clients and answers 429 once you push past it; a first version of
    this function treated 429 like a hard failure after 3 quick tries and silently lost 43 of 85 articles.
    Retry-After is honoured when present, and the base pause between successful requests is deliberately
    ~1 s -- this fetches fewer than 150 documents in total, so there is no reason to go faster.
    """
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read().decode("utf-8", errors="replace")
            time.sleep(pause)                      # be a polite client
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                print(f"    HTTP {e.code} (gave up after {retries} tries) {url}")
                return None
            wait = float(e.headers.get("Retry-After") or 0) if e.headers else 0.0
            wait = max(wait, 2.0 * (2 ** attempt))  # 2, 4, 8, 16, 32 s
            if e.code == 429:
                print(f"    HTTP 429 rate-limited, backing off {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:                      # noqa: BLE001 -- network flake, retry
            if attempt == retries - 1:
                print(f"    {type(e).__name__} {url}")
                return None
            time.sleep(2.0 * (2 ** attempt))
    return None


def wikipedia_extract(title: str) -> tuple[str, str] | None:
    """Plain-text extract of one article. One title per request: the API lowers `exlimit` to 1 for whole
    -article extracts, so batching silently returns empty extracts for all but the first title."""
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2", "prop": "extracts",
        "explaintext": "1", "exlimit": "1", "redirects": "1", "titles": title,
    })
    raw = fetch(f"https://en.wikipedia.org/w/api.php?{q}")
    if not raw:
        return None
    try:
        pages = json.loads(raw)["query"]["pages"]
    except (KeyError, json.JSONDecodeError):
        return None
    for p in pages:
        if p.get("missing") or not p.get("extract"):
            return None
        return p["title"], p["extract"]
    return None


# ------------------------------------------------------------------------------------------------ cleaning

# Sections that are navigation/citation residue, not prose.
DROP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes", "citations",
    "bibliography", "sources", "footnotes", "gallery", "explanatory notes", "works cited",
    "general sources", "general and cited references", "additional resources", "further resources",
}

WIKI_SECTION_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def _normalize_unicode(t: str) -> str:
    # ethereum.org Markdown carries HTML entities (`&lt;5GB`, `a &amp; b`) that must not reach the prose.
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", " - ").replace(" ", " ")
    t = re.sub(r"[​-‏‪-‮﻿]", "", t)       # zero-width / bidi marks
    return t


def split_wikipedia(title: str, extract: str) -> list[tuple[str, str]]:
    """-> [(section_path, prose)] with boilerplate sections removed. The lead (before the first heading)
    becomes the section "Introduction"."""
    text = _normalize_unicode(extract)
    out: list[tuple[str, str]] = []
    matches = list(WIKI_SECTION_RE.finditer(text))
    lead = text[: matches[0].start()] if matches else text
    if lead.strip():
        out.append((f"{title} - Introduction", lead.strip()))
    for i, m in enumerate(matches):
        name = m.group(2).strip()
        body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        if name.lower().strip(" :") in DROP_SECTIONS:
            continue
        if body.strip():
            out.append((f"{title} - {name}", body.strip()))
    return out


ADOC_BLOCK_DELIM = re.compile(r"^(-{4,}|={4,}|\*{4,}|\.{4,}|_{4,}|\+{4,}|/{4,})\s*$")
ADOC_SECTION_RE = re.compile(r"^(={1,6})\s+(.+?)\s*$", re.M)


def clean_asciidoc(raw: str) -> str:
    """Strip AsciiDoc markup, code/table/admonition blocks and index entries, keeping prose.

    "Mastering Bitcoin" uses `(((index, terms)))` very heavily and embeds large code/console listings; both
    would otherwise dominate the text and teach the model nothing explanatory.
    """
    raw = _normalize_unicode(raw)
    raw = re.sub(r"\(\(\(.*?\)\)\)", "", raw, flags=re.S)           # index entries
    # `pass:[...]` passthrough macros embed raw HTML (the whitepaper's byline carries an <a href=...><em>
    # link). Keep the inner words, drop the macro and the tags.
    raw = re.sub(r"pass:[a-z]*\[(.*?)\]", r" \1 ", raw, flags=re.S)
    raw = re.sub(r"</?[a-zA-Z][^>]*>", "", raw)                     # residual inline HTML
    raw = re.sub(r"^\[\[.*?\]\]\s*$", "", raw, flags=re.M)          # anchors
    raw = re.sub(r"^:[\w!-]+:.*$", "", raw, flags=re.M)             # attribute entries
    raw = re.sub(r"^(include|image|video|ifdef|ifndef|endif)::.*$", "", raw, flags=re.M)
    raw = re.sub(r"footnote:\[(.*?)\]", r" (\1)", raw, flags=re.S)
    raw = re.sub(r"<<[^>]*?,\s*([^>]*?)>>", r"\1", raw)             # xref with label
    raw = re.sub(r"<<([^>,]*?)>>", "", raw)                         # bare xref
    raw = re.sub(r"(?:https?|link):[^\s\[]*\[(.*?)\]", r"\1", raw, flags=re.S)
    raw = re.sub(r"\bhttps?://\S+", "", raw)

    lines, keep, in_block, in_table = raw.split("\n"), [], False, False
    skip_next_block = False
    for ln in lines:
        s = ln.rstrip()
        if s.startswith("|==="):
            in_table = not in_table
            continue
        if in_table:
            continue
        if ADOC_BLOCK_DELIM.match(s):
            if in_block:
                in_block = False
            else:
                in_block = True if skip_next_block else in_block
                if not skip_next_block:
                    # an undecorated delimited block: treat as literal/code and skip it too
                    in_block = True
            skip_next_block = False
            continue
        if in_block:
            continue
        if re.match(r"^\[(source|listing|literal|quote|verse|NOTE|TIP|WARNING|CAUTION|IMPORTANT|"
                    r"role|cols|options|width|frame|grid|caption|id)[,\]=]", s, re.I):
            skip_next_block = True
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        keep.append(s)
    text = "\n".join(keep)

    text = re.sub(r"`{1,3}([^`]*?)`{1,3}", r"\1", text)             # inline code
    # `+literal+` -> literal, but only short spans with no internal whitespace: a greedy `\+(...)\+` turned
    # the arithmetic "consume +t+ + +k+ + 2 items" into "consume t  k  2 items", losing the operators.
    text = re.sub(r"(?<!\+)\+([^\s+][^+\n]{0,40}?)\+(?!\+)", r"\1", text)
    text = re.sub(r"~([^~\s][^~\n]{0,30}?)~", r"\1", text)          # subscript:   F~sig~  -> Fsig
    text = re.sub(r"\^([^\^\s][^\^\n]{0,30}?)\^", r"^\1", text)     # superscript: 2^256^  -> 2^256
    text = re.sub(r"\*{1,2}([^*\n]+?)\*{1,2}", r"\1", text)         # bold
    text = re.sub(r"(?<![\w_])_([^_\n]+?)_(?![\w_])", r"\1", text)  # italic
    text = re.sub(r"^[\s]*[*.]{1,5}\s+", "", text, flags=re.M)      # list bullets
    text = re.sub(r"^\s*(NOTE|TIP|WARNING|CAUTION|IMPORTANT):\s*", "", text, flags=re.M)
    return text


def split_asciidoc(title: str, raw: str) -> list[tuple[str, str]]:
    text = clean_asciidoc(raw)
    out: list[tuple[str, str]] = []
    matches = list(ADOC_SECTION_RE.finditer(text))
    if not matches:
        return [(title, text.strip())] if text.strip() else []
    for i, m in enumerate(matches):
        name = m.group(2).strip()
        body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        if name.lower().strip(" :") in DROP_SECTIONS:
            continue
        if body.strip():
            out.append((f"{title} - {name}", body.strip()))
    return out


MW_SECTION_RE = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def clean_mediawiki(raw: str) -> str:
    """Strip MediaWiki markup from a BIP: code/pre blocks, tables, links and inline emphasis.

    BIPs are specs, so they carry large `<pre>` reference-implementation blocks and wire-format tables that
    are mostly symbols -- exactly what `_quality_ok`'s digit/letter-ratio filters would reject downstream,
    but cheaper to remove here than to let them split good prose into unusable fragments.
    """
    raw = _normalize_unicode(raw)
    raw = re.sub(r"<(pre|source|syntaxhighlight|code|nowiki)[^>]*>.*?</\1>", "", raw, flags=re.S | re.I)
    raw = re.sub(r"\{\|.*?\|\}", "", raw, flags=re.S)                      # tables
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"</?[a-zA-Z][^>]*>", "", raw)                            # stray html
    raw = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", raw)            # [[target|label]] -> label
    raw = re.sub(r"\[(?:https?|ftp)://\S+\s+([^\]]*)\]", r"\1", raw)       # [url label] -> label
    raw = re.sub(r"\[(?:https?|ftp)://\S+\]", "", raw)
    raw = re.sub(r"\bhttps?://\S+", "", raw)
    raw = re.sub(r"'{2,5}", "", raw)                                       # ''italic'' / '''bold'''
    raw = re.sub(r"^\s*[*#:;]+\s*", "", raw, flags=re.M)                   # list / indent markers
    raw = re.sub(r"^\s*\|.*$", "", raw, flags=re.M)                        # leftover table rows
    return raw


def split_mediawiki(title: str, raw: str) -> list[tuple[str, str]]:
    text = clean_mediawiki(raw)
    out: list[tuple[str, str]] = []
    matches = list(MW_SECTION_RE.finditer(text))
    lead = text[: matches[0].start()] if matches else text
    if lead.strip():
        out.append((f"{title} - Introduction", lead.strip()))
    for i, m in enumerate(matches):
        name = m.group(2).strip()
        body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        if name.lower().strip(" :") in DROP_SECTIONS:
            continue
        if body.strip():
            out.append((f"{title} - {name}", body.strip()))
    return out


def repo_tarball_members(cache: Path, repo: str, subdir: str, suffixes: tuple[str, ...],
                         min_bytes: int, allow_fetch: bool) -> list[tuple[str, str]]:
    """-> [(filename, text)] for files under `subdir` in a GitHub repo, via its source tarball.

    Deliberately NOT the contents API: unauthenticated GitHub API calls are capped at 60/hour, so listing
    a directory and then pulling N raw files both rate-limits (HTTP 403, which silently yields zero
    documents) and costs N+1 requests. `codeload.github.com/<repo>/tar.gz/refs/heads/<branch>` is one
    request, is not part of the API quota, and is cached here like every other download.
    """
    import io
    import tarfile

    tar_path = cache / (re.sub(r"[^A-Za-z0-9._-]+", "_", repo) + ".tar.gz")
    if not tar_path.exists():
        if not allow_fetch:
            return []
        for branch in ("master", "main"):
            url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                tar_path.write_bytes(data)
                print(f"    downloaded {repo}@{branch} ({len(data) / 1e6:.1f} MB)")
                break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"    HTTP {e.code} for {url}")
            except Exception as e:                                   # noqa: BLE001
                print(f"    {type(e).__name__} for {url}")
        if not tar_path.exists():
            return []

    out: list[tuple[str, str]] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.size < min_bytes:
                continue
            # tarball paths are "<repo>-<branch>/<path>"; compare on the path inside the repo
            inner = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if subdir and not inner.startswith(subdir.rstrip("/") + "/"):
                continue
            name = Path(inner).name
            if not name.endswith(suffixes):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            out.append((name, fh.read().decode("utf-8", errors="replace")))
    return sorted(out)


MD_SECTION_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


def clean_markdown(raw: str) -> str:
    """Strip YAML frontmatter, code fences, JSX/MDX components, links and tables from ethereum.org docs."""
    raw = _normalize_unicode(raw)
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)         # frontmatter
    raw = re.sub(r"```.*?```", "", raw, flags=re.S)                 # fenced code
    raw = re.sub(r"~~~.*?~~~", "", raw, flags=re.S)
    raw = re.sub(r"\{/\*.*?\*/\}", "", raw, flags=re.S)             # MDX comments
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"^import\s+.*$", "", raw, flags=re.M)
    raw = re.sub(r"^export\s+.*$", "", raw, flags=re.M)
    raw = re.sub(r"<[A-Z][\w.]*(?:\s[^>]*?)?/>", "", raw, flags=re.S)        # self-closing components
    raw = re.sub(r"</?[A-Z][\w.]*(?:\s[^>]*?)?>", "", raw, flags=re.S)       # component open/close
    raw = re.sub(r"</?[a-z][\w]*(?:\s[^>]*?)?>", "", raw)                    # stray html
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)                           # images
    raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)                       # links -> label
    raw = re.sub(r"^\s*\|.*\|\s*$", "", raw, flags=re.M)                     # table rows
    raw = re.sub(r"`{1,3}([^`]*?)`{1,3}", r"\1", raw)
    raw = re.sub(r"\*{1,2}([^*\n]+?)\*{1,2}", r"\1", raw)
    raw = re.sub(r"^\s*[-*+]\s+", "", raw, flags=re.M)
    raw = re.sub(r"^\s*\d+\.\s+", "", raw, flags=re.M)
    raw = re.sub(r"^\s*>\s?", "", raw, flags=re.M)
    raw = re.sub(r"\s*\{#[\w-]+\}\s*$", "", raw, flags=re.M)        # heading anchors: "## Gas {#gas}"
    return raw


def split_markdown(title: str, raw: str) -> list[tuple[str, str]]:
    text = clean_markdown(raw)
    out: list[tuple[str, str]] = []
    matches = list(MD_SECTION_RE.finditer(text))
    lead = text[: matches[0].start()] if matches else text
    if lead.strip():
        out.append((f"{title} - Introduction", lead.strip()))
    for i, m in enumerate(matches):
        name = m.group(2).strip()
        body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        if name.lower().strip(" :") in DROP_SECTIONS:
            continue
        if body.strip():
            out.append((f"{title} - {name}", body.strip()))
    return out


# --------------------------------------------------------------------------------------- quality + chunking

MIN_CHARS, MAX_CHARS, MIN_WORDS = 400, 6000, 60


def tidy(text: str) -> str:
    lines = []
    for ln in text.split("\n"):
        s = re.sub(r"[ \t]+", " ", ln).strip()
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _quality_ok(text: str) -> tuple[bool, str]:
    """Reject anything that is not explanatory prose. The thresholds target the specific junk this corpus
    must not contain: price/market tables (digit-heavy), stat dumps and listicles (short lines, few verbs),
    and citation residue."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return False, "too_few_words"
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    if letters == 0 or digits / max(len(text), 1) > 0.12:
        return False, "digit_heavy"           # price series, OHLC tables, date dumps
    if letters / max(len(text), 1) < 0.60:
        return False, "low_letter_ratio"      # markup/symbol residue
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and sum(len(l) for l in lines) / len(lines) < 40:
        return False, "line_fragments"        # table/list residue rather than sentences
    sentences = re.split(r"[.!?]\s", text)
    if len(sentences) < 3:
        return False, "too_few_sentences"
    if len(words) / max(len(set(w.lower() for w in words)), 1) > 6.0:
        return False, "repetitive"
    low = text.lower()
    if low.count("retrieved ") + low.count("archived from the original") >= 3:
        return False, "citation_residue"
    return True, "ok"


def chunk(text: str) -> list[str]:
    """Split an over-long section on paragraph boundaries into MAX_CHARS-ish documents."""
    text = tidy(text)
    if len(text) <= MAX_CHARS:
        return [text]
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 > MAX_CHARS and cur:
            out.append(cur.strip())
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        out.append(cur.strip())
    return out


# ------------------------------------------------------------------------------------------------- dedup

def _norm_for_hash(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


class Dedup:
    """Exact dedup on normalized text, plus near-dup detection via shared word 8-grams.

    Wikipedia articles repeat each other's lead paragraphs and "Mastering Bitcoin" restates the whitepaper,
    so near-duplicates are common and would otherwise be seen many times per epoch.
    """

    SHINGLE, SAMPLES, THRESHOLD = 8, 24, 0.5

    def __init__(self):
        self.exact: set[str] = set()
        self.index: dict[int, list[int]] = {}
        self.doc_shingles: list[set[int]] = []

    def _shingles(self, text: str) -> list[int]:
        w = _norm_for_hash(text).split()
        grams = [" ".join(w[i:i + self.SHINGLE]) for i in range(max(0, len(w) - self.SHINGLE + 1))]
        hs = sorted({int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16) for g in grams})
        if len(hs) <= self.SAMPLES:
            return hs
        step = len(hs) / self.SAMPLES
        return [hs[int(i * step)] for i in range(self.SAMPLES)]

    def add(self, text: str) -> tuple[bool, str]:
        h = hashlib.blake2b(_norm_for_hash(text).encode(), digest_size=16).hexdigest()
        if h in self.exact:
            return False, "exact_duplicate"
        sh = self._shingles(text)
        if sh:
            hits: dict[int, int] = {}
            for s in sh:
                for d in self.index.get(s, ()):
                    hits[d] = hits.get(d, 0) + 1
            for d, n in hits.items():
                union = len(self.doc_shingles[d] | set(sh))
                if union and n / union >= self.THRESHOLD:
                    return False, "near_duplicate"
        self.exact.add(h)
        idx = len(self.doc_shingles)
        self.doc_shingles.append(set(sh))
        for s in sh:
            self.index.setdefault(s, []).append(idx)
        return True, "ok"


# --------------------------------------------------------------------------------------------------- main

def default_out() -> str:
    return "/workspace/ppx_text_corpus" if Path("/workspace").is_dir() else "./ppx_text_corpus"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=default_out(), help=f"corpus directory (default: {default_out()})")
    ap.add_argument("--cache", default=None,
                    help="raw-download cache (default: <out>-cache, a SIBLING of the corpus). It must stay "
                         "outside the corpus directory: distill/data/text_dataset.py walks --text-data "
                         "recursively for .txt/.md/.jsonl/.json, so a cache inside it would be scanned on "
                         "every run (91 MediaWiki dumps, ~1.6 MB of JSON parsed for nothing).")
    ap.add_argument("--no-fetch", action="store_true", help="use only what is already in the cache")
    ap.add_argument("--sources", nargs="*",
                    default=["wikipedia", "mastering_bitcoin", "ethereum_org", "bips", "eips"],
                    choices=["wikipedia", "mastering_bitcoin", "ethereum_org", "bips", "eips"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out.parent / (out.name + "-cache")
    cache.mkdir(parents=True, exist_ok=True)
    if cache.resolve() == out.resolve() or out.resolve() in cache.resolve().parents:
        raise SystemExit(f"--cache {cache} is inside the corpus directory {out}; the loader would scan it")

    def cached(key: str, url: str) -> str | None:
        f = cache / (re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:150] + ".raw")
        if f.exists():
            return f.read_text(encoding="utf-8", errors="replace")
        if args.no_fetch:
            return None
        body = fetch(url)
        if body:
            f.write_text(body, encoding="utf-8")
        return body

    # ---- 1. download + split into (source, topic, title, section_text) ------------------------------------
    raw_sections: list[tuple[str, str, str, str]] = []
    fetched = {k: 0 for k in LICENSES}
    missing: list[str] = []

    if "wikipedia" in args.sources:
        print("fetching Wikipedia ...")
        for topic, titles in WIKIPEDIA.items():
            for t in titles:
                f = cache / (re.sub(r"[^A-Za-z0-9._-]+", "_", "wiki_" + t)[:150] + ".json")
                if f.exists():
                    got = json.loads(f.read_text(encoding="utf-8"))
                elif args.no_fetch:
                    missing.append(f"wikipedia:{t}")
                    continue
                else:
                    res = wikipedia_extract(t)
                    if res is None:
                        missing.append(f"wikipedia:{t}")
                        print(f"    MISSING {t}")
                        continue
                    got = {"title": res[0], "extract": res[1]}
                    f.write_text(json.dumps(got), encoding="utf-8")
                fetched["wikipedia"] += 1
                for name, body in split_wikipedia(got["title"], got["extract"]):
                    raw_sections.append(("wikipedia", topic, name, body))

    if "mastering_bitcoin" in args.sources:
        print("fetching Mastering Bitcoin ...")
        for topic, files in MASTERING_BITCOIN.items():
            for fn in files:
                body = cached("mb_" + fn, MASTERING_BITCOIN_RAW + fn)
                if not body:
                    missing.append(f"mastering_bitcoin:{fn}")
                    continue
                fetched["mastering_bitcoin"] += 1
                stem = fn.replace(".adoc", "").replace("_", " ")
                for name, sec in split_asciidoc(f"Mastering Bitcoin: {stem}", body):
                    raw_sections.append(("mastering_bitcoin", topic, name, sec))

    if "ethereum_org" in args.sources:
        print("fetching ethereum.org docs ...")
        seen_paths: set[str] = set()
        for topic, paths in ETHEREUM_ORG.items():
            for p in paths:
                key = p.strip("/")
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                body = cached("eth_" + key, f"{ETHEREUM_ORG_RAW}{key}/index.md")
                if not body:
                    missing.append(f"ethereum_org:{key}")
                    continue
                fetched["ethereum_org"] += 1
                for name, sec in split_markdown(f"ethereum.org: {key}", body):
                    raw_sections.append(("ethereum_org", topic, name, sec))

    if "bips" in args.sources:
        print("fetching Bitcoin Improvement Proposals ...")
        members = repo_tarball_members(cache, BIPS_REPO, "", (".mediawiki", ".md"),
                                       BIP_MIN_BYTES, not args.no_fetch)
        print(f"    {len(members)} BIPs above {BIP_MIN_BYTES} bytes")
        for fn, body in members:
            fetched["bips"] += 1
            stem = fn.rsplit(".", 1)[0]
            split = split_mediawiki if fn.endswith(".mediawiki") else split_markdown
            for name, sec in split(f"BIP: {stem}", body):
                raw_sections.append(("bips", "bitcoin", name, sec))

    if "eips" in args.sources:
        print("fetching Ethereum Improvement Proposals ...")
        members = repo_tarball_members(cache, EIPS_REPO, "EIPS", (".md",),
                                       EIP_MIN_BYTES, not args.no_fetch)
        print(f"    {len(members)} EIPs above {EIP_MIN_BYTES} bytes")
        for fn, body in members:
            fetched["eips"] += 1
            stem = fn.rsplit(".", 1)[0]
            for name, sec in split_markdown(f"EIP: {stem}", body):
                raw_sections.append(("eips", "ethereum", name, sec))

    print(f"\nsections before filtering: {len(raw_sections)}")

    # ---- 2. chunk, quality filter, dedup -----------------------------------------------------------------
    dedup = Dedup()
    rejected: dict[str, int] = {}
    docs: list[dict] = []
    for source, topic, title, body in raw_sections:
        for piece in chunk(body):
            if len(piece) < MIN_CHARS:
                rejected["too_short"] = rejected.get("too_short", 0) + 1
                continue
            ok, why = _quality_ok(piece)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            ok, why = dedup.add(piece)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            docs.append({"text": piece, "title": title, "topic": topic,
                         "source": source, "license": LICENSES[source]})

    # ---- 3. write JSONL shards, one per source ----------------------------------------------------------
    shard_name = {"wikipedia": "01_wikipedia.jsonl",
                  "mastering_bitcoin": "02_mastering_bitcoin.jsonl",
                  "ethereum_org": "03_ethereum_org_docs.jsonl",
                  "bips": "04_bitcoin_bips.jsonl",
                  "eips": "05_ethereum_eips.jsonl"}
    written: dict[str, dict] = {}
    for source, fname in shard_name.items():
        rows = [d for d in docs if d["source"] == source]
        if not rows:
            continue
        path = out / fname
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for d in rows:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        written[fname] = {"documents": len(rows), "characters": sum(len(d["text"]) for d in rows),
                          "bytes": path.stat().st_size, "license": LICENSES[source]}

    # Remove every shard this run did not just write. Two ways a stale shard survives otherwise: a shard
    # gets RENAMED (01_wikipedia_crypto -> 01_wikipedia), or a source yields zero rows this run (its
    # listing 403'd) and `continue` skips rewriting it. Both leave old documents in the corpus, and the
    # self-check below caught each of them in turn (+894, then +2754).
    for stale in sorted(out.glob("*.jsonl")):
        if stale.name not in written:
            print(f"  removing stale shard: {stale.name}")
            stale.unlink()

    by_topic: dict[str, dict] = {}
    for d in docs:
        e = by_topic.setdefault(d["topic"], {"documents": 0, "characters": 0})
        e["documents"] += 1
        e["characters"] += len(d["text"])

    total_chars = sum(len(d["text"]) for d in docs)
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "tools/build_text_corpus.py",
        "format": "JSONL, one document per line, 'text' key -- read by distill/data/text_dataset.py",
        "sources": {s: {"files_or_articles": n, "license": LICENSES[s]} for s, n in fetched.items() if n},
        "totals": {"documents": len(docs), "characters": total_chars,
                   "words": sum(len(d["text"].split()) for d in docs)},
        "files": written,
        "by_topic": dict(sorted(by_topic.items())),
        "rejected": dict(sorted(rejected.items(), key=lambda kv: -kv[1])),
        "missing_sources": missing,
        "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "min_words": MIN_WORDS,
                    "max_digit_ratio": 0.12, "min_letter_ratio": 0.60, "min_avg_line_chars": 40,
                    "near_dup_shingle": Dedup.SHINGLE, "near_dup_threshold": Dedup.THRESHOLD},
    }
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    notice = f"""# Text corpus for PersonaPlex student text-only distillation

Built by `tools/build_text_corpus.py` on {manifest['built']}.
Format: JSONL, one document per line with a `"text"` key -- exactly what
`distill/data/text_dataset.py` reads. Extra keys (`title`, `topic`, `source`, `license`) are
provenance only and are ignored by the loader.

Totals: {len(docs):,} documents, {total_chars:,} characters.

## Sources and licenses

1. **English Wikipedia** -- Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).
   Retrieved via the MediaWiki `extracts` API. Articles are listed in `corpus_manifest.json`
   and in each document's `title` field. https://en.wikipedia.org
2. **Mastering Bitcoin, 3rd edition** -- Andreas M. Antonopoulos and David A. Harding,
   Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).
   https://github.com/bitcoinbook/bitcoinbook  (Appendix A is the Bitcoin whitepaper.)
3. **ethereum.org developer documentation** -- Creative Commons Attribution 4.0 (CC BY 4.0).
   https://github.com/ethereum/ethereum-org-website

All three are used unmodified in substance; the builder strips markup, code listings, tables and
citation sections, and splits articles into section-sized documents. CC BY-SA requires that
derivative distributions carry the same license and attribution -- keep this file alongside the
corpus, and note that a model trained on it is generally not considered a derivative work of the
text, but redistributing the corpus itself is.

This file is named `NOTICE` with no extension on purpose: `distill/data/text_dataset.py` treats
`.md` and `.txt` files as training documents, so a `NOTICE.md` here would be trained on.

## Deliberately excluded

Price and market data, OHLC series, charts, social-media posts, forum scrapes, SEO/affiliate
content and noisy scraped crypto datasets. The quality filter in the builder rejects digit-heavy
text, line fragments, listicles and citation residue; see `corpus_manifest.json -> rejected`.
"""
    # Deliberately NO .md/.txt extension: `TEXT_SUFFIXES` in distill/data/text_dataset.py includes ".md",
    # so a `NOTICE.md` sitting beside the shards is read as a TRAINING DOCUMENT -- the attribution text would
    # end up in the corpus. (`corpus_manifest.json` is safe: it is a .json dict with no recognized text key,
    # so `_iter_documents` yields nothing from it. The self-check below enforces both.)
    (out / "NOTICE").write_text(notice, encoding="utf-8")
    for stale in ("NOTICE.md", "NOTICE.txt"):
        if (out / stale).exists():
            (out / stale).unlink()

    # ---- 4. self-check against the REAL loader ----------------------------------------------------------
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent),
                    str(Path(__file__).resolve().parent.parent / "moshi")]
    from distill.data.text_dataset import _iter_documents
    seen = list(_iter_documents([str(out)]))
    if len(seen) != len(docs):
        extra = len(seen) - len(docs)
        raise SystemExit(
            f"SELF-CHECK FAILED: distill/data/text_dataset.py reads {len(seen)} documents from {out} but "
            f"{len(docs)} were written ({extra:+d}). Something in the corpus directory that is not a data "
            f"shard is being picked up as training text -- check for stray .txt/.md/.jsonl/.json files.")
    print(f"\nself-check: the real loader reads exactly {len(seen):,} documents from {out} -- matches.")

    print(f"\n{'=' * 78}\ncorpus written to {out}")
    print(f"  documents : {len(docs):,}")
    print(f"  characters: {total_chars:,}")
    for f, s in written.items():
        print(f"  {f:34s} {s['documents']:>6,} docs  {s['characters']:>10,} chars")
    print(f"\nrejected: {json.dumps(manifest['rejected'])}")
    if missing:
        print(f"missing ({len(missing)}): {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    print("manifest -> corpus_manifest.json ; attribution -> NOTICE (no extension, on purpose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
