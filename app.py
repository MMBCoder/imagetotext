import streamlit as st
from agents.orchestrator import run_sales_meeting_prep

st.set_page_config(page_title='SalesPrep', layout='wide')

st.title('SalesPrep')
st.markdown('Prepare FMCG sales representatives before customer meetings using AI insights.')

st.sidebar.header('API Configuration')

openai_api_key = st.sidebar.text_input(
    'OpenAI API Key',
    type='password'
)

tavily_api_key = st.sidebar.text_input(
    'Tavily API Key',
    type='password'
)

model_name = st.sidebar.selectbox(
    'LLM Model',
    ['gpt-5', 'gpt-5-mini'],
    index=0
)

company = st.text_input('Company/Product', 'Cadbury Dairy Milk')
competitor = st.text_input('Competitor', 'Nestle')

if st.button('Generate Intelligence Report'):

    if not openai_api_key or not tavily_api_key:
        st.error('Please provide API keys in the sidebar.')
    else:
        with st.spinner('Running AI agents...'):
            result = run_sales_meeting_prep(
                company=company,
                competitor=competitor,
                openai_api_key=openai_api_key,
                tavily_api_key=tavily_api_key,
                model_name=model_name,
            )

        st.subheader('Executive Summary')
        st.write(result['summary'])

        st.subheader('Market Insights')
        st.write(result['market_insights'])

        st.subheader('Customer Sentiment')
        st.write(result['sentiment'])

        st.subheader('Financial Analysis')
        st.write(result['finance'])

        st.subheader('Recommendations')
        st.write(result['recommendations'])
