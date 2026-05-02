# ExecLab AI Business Document

## One-Sentence Business Case
ExecLab AI is an agentic execution-analysis copilot for small funds, student investment funds, fintech product teams, and quant students who need to compare benchmark execution approaches and explain the tradeoffs without buying a full institutional OMS/EMS analytics stack.

## The User
The primary user is a small-fund analyst, student investment fund trader, junior execution trader, or fintech product team member who needs to understand how execution choices change cost, completion risk, and benchmark slippage.

The secondary user is an instructor, mentor, or PM reviewing how a junior trader thinks. ExecLab AI creates a repeatable environment where the same order can be tested under multiple algorithms, risk assumptions, and custom constraints, then explained by specialized agents.

## The Problem
Users often rely on static spreadsheets, one-off notebooks, or high-level broker descriptions of execution algorithms. That creates four problems:

- Black-box algorithm vocabulary: users know the names but cannot see the schedule mechanics.
- Weak pre-trade reasoning: users may not connect volume, volatility, spread proxy, beta risk, peer movement, and order size.
- Poor explanation quality: even when metrics are computed, the user still needs to turn tables into a defensible memo.
- No safe customization loop: users want to say "keep participation under 10%" or "get 50% done by 11:00" without relying on brittle hard-coded parsing.

## The Product
ExecLab AI live-fetches public intraday OHLCV bars, runs deterministic schedule and TCA tools, then uses Google ADK and Vertex Gemini agents to explain the results. The product compares TWAP, VWAP, POV, implementation-shortfall style schedules, and an agent-designed custom schedule.

## The Economics
ExecLab AI would be sold as lightweight SaaS research software:

- Free/student tier: limited backtests for education and demos.
- Student/pro tier: about $10-$49 per month for more runs, saved scenarios, and richer agent memos.
- Team tier: about $199 per month for shared workspaces, templates, and training use.

The cost structure is favorable because the numerical work is cheap Python computation, the default market data provider has no API-key cost, and the LLM work is limited to compact structured context rather than raw bar data.

## Why The Technical Choices Fit
Google ADK gives the product real multi-agent orchestration. Python tools keep the math auditable. Pydantic schemas keep agent outputs reliable. The CustomAlgoPlannerAgent turns natural-language desk constraints into structured execution parameters. Cloud Run provides a public URL and low operating cost.

## Deployment Proof

- Live app: https://execlab-ai-q7smatrnpa-uc.a.run.app
- Cloud Run service: execlab-ai
- Verified revision: execlab-ai-00005-b5j
- Full ADK smoke: ADK_FULL_SMOKE_OK with 19 agent reports and Gemini model gemini-2.5-flash-lite
