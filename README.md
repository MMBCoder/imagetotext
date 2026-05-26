# AI Multi-Agent Sales Copilot

Enterprise-grade AI sales intelligence application built using Streamlit, GPT-5, Tavily, LangGraph, and financial analytics.

## Features

- Multi-agent AI orchestration
- Tavily web intelligence
- FMCG customer review analysis
- Nestle financial comparison
- GPT-5 executive summary generation
- Sales recommendations engine
- Streamlit dashboard
- Forecast-ready architecture

## Architecture

- Research Agent
- Sentiment Agent
- Finance Agent
- Recommendation Agent
- Executive Summary Agent
- Orchestrator Agent

## Tech Stack

- Streamlit
- GPT-5 API
- Tavily API
- LangGraph
- yFinance
- Pandas
- Plotly
- Prophet

## Setup

### Clone Repository

```bash
git clone https://github.com/MMBCoder/imagetotext.git
cd imagetotext
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Copy `.env.example` into `.env`

```bash
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
MODEL_NAME=gpt-5
```

### Run Application

```bash
streamlit run app.py
```

## Folder Structure

```text
agents/
services/
tools/
app.py
requirements.txt
README.md
.env.example
```

## Future Enhancements

- CRM integrations
- SAP integration
- Forecasting dashboards
- Voice-enabled sales assistant
- PowerPoint generation
- Territory analytics
- Vector database memory

## License

MIT
