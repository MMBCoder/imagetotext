from agents.research_agent import research_company
from agents.finance_agent import analyze_finance
from agents.sentiment_agent import analyze_sentiment
from agents.recommendation_agent import generate_recommendations
from services.llm_service import summarize_report


def run_sales_meeting_prep(
    company,
    competitor,
    openai_api_key,
    tavily_api_key,
    model_name,
):

    market_data = research_company(company, tavily_api_key)

    finance_data = analyze_finance(competitor)

    sentiment_data = analyze_sentiment(company)

    recommendations = generate_recommendations(
        market_data,
        finance_data,
        sentiment_data,
    )

    summary = summarize_report(
        company,
        market_data,
        finance_data,
        sentiment_data,
        recommendations,
        openai_api_key,
        model_name,
    )

    return {
        'summary': summary,
        'market_insights': market_data,
        'finance': finance_data,
        'sentiment': sentiment_data,
        'recommendations': recommendations,
    }
