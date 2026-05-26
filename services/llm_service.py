import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def summarize_report(company, market, finance, sentiment, recommendations):
    prompt = f'''
    Prepare a detailed FMCG sales meeting preparation summary.

    Company: {company}

    Market Data:
    {market}

    Finance Data:
    {finance}

    Sentiment:
    {sentiment}

    Recommendations:
    {recommendations}
    '''

    response = client.chat.completions.create(
        model=os.getenv('MODEL_NAME', 'gpt-5'),
        messages=[
            {
                'role': 'system',
                'content': 'You are a senior FMCG sales strategist.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response.choices[0].message.content
