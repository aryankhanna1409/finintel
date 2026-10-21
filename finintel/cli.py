"""Command-line interface.

Examples::

    finintel report TGT --peers WMT COST DG KR --out reports/           # offline fixtures
    finintel report TGT --peers WMT COST DG KR --live --user-agent "Acme Research ops@acme.com"
    finintel ask "What did Target say about tariffs?" --ticker TGT
    finintel graph --cypher graph.cypher
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .agents.orchestrator import FinancialIntelligencePipeline, PipelineConfig
from .reporting.render import to_json, to_markdown


def _pipeline(args: argparse.Namespace) -> FinancialIntelligencePipeline:
    cfg = PipelineConfig(embedder=args.embedder, reranker=args.reranker, use_llm=False if args.no_llm else None, model=args.model)
    if args.live:
        if not args.user_agent:
            sys.exit("--live requires --user-agent 'Name contact@email' (SEC fair-access policy)")
        if not getattr(args, "ticker", None):
            sys.exit("--live needs --ticker (and optionally --peers) to know which issuers to fetch")
        tickers = [args.ticker, *(args.peers or [])]
        return FinancialIntelligencePipeline.from_live(tickers, args.user_agent, cfg)
    return FinancialIntelligencePipeline.from_fixtures(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finintel", description="Financial Intelligence & RAG system")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--live", action="store_true", help="fetch from SEC EDGAR / Stooq / FRED instead of offline fixtures")
        p.add_argument("--user-agent", help="SEC User-Agent with contact email (required with --live)")
        p.add_argument("--embedder", default=None, help="'hashing' (default) or a sentence-transformers model id")
        p.add_argument("--reranker", default=None, help="'features' (default) or a cross-encoder model id")
        p.add_argument("--model", default=None, help="Claude model id (default claude-opus-5-5)")
        p.add_argument("--no-llm", action="store_true", help="force the deterministic narrative")

    rp = sub.add_parser("report", help="generate a Strategic Advisory Report")
    rp.add_argument("ticker")
    rp.add_argument("--peers", nargs="*", help="3-5 peer tickers (default: graph-selected)")
    rp.add_argument("--out", default="reports", help="output directory")
    common(rp)

    qp = sub.add_parser("ask", help="grounded, quote-only question answering over the corpus")
    qp.add_argument("question")
    qp.add_argument("--ticker", default=None)
    qp.set_defaults(peers=None)
    common(qp)

    gp = sub.add_parser("graph", help="print graph stats and optionally export Cypher for Neo4j")
    gp.add_argument("--cypher", default=None, help="write a Cypher script to this path")
    gp.add_argument("--ticker", default="TGT")
    gp.set_defaults(peers=None)
    common(gp)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pipe = _pipeline(args)

    if args.cmd == "report":
        rep = pipe.advise(args.ticker, args.peers)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        md, js = out / f"{rep.target_ticker}_advisory_{rep.as_of}.md", out / f"{rep.target_ticker}_advisory_{rep.as_of}.json"
        md.write_text(to_markdown(rep), encoding="utf-8")
        js.write_text(to_json(rep), encoding="utf-8")
        print(f"narrative engine: {rep.narrative_engine}; grounding rate {rep.grounding.get('grounding_rate')}")
        print(f"wrote {md}\nwrote {js}")
        print("\n" + rep.executive_summary)
        return 0
    if args.cmd == "ask":
        ans = pipe.ask(args.question, args.ticker)
        if ans.abstained:
            print(f"ABSTAINED: {ans.abstain_reason}")
            return 0
        for c in ans.claims:
            print(f"- {c.claim} [{', '.join(c.evidence_ids)}]")
        print()
        for c in ans.citations:
            print(c.render())
        return 0
    if args.cmd == "graph":
        print(pipe.kg.stats().model_dump_json(indent=2))
        if args.cypher:
            from .graph.neo4j_export import to_cypher_script

            Path(args.cypher).write_text(to_cypher_script(pipe.kg), encoding="utf-8")
            print(f"wrote {args.cypher}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
