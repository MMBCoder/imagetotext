from openai import OpenAI


def summarize_report(
    company,
    market,
    finance,
    sentiment,
    recommendations,
    openai_api_key,
    model_name,
):

    client = OpenAI(api_key=openai_api_key)

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
        model=model_name,
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
