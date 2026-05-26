import streamlit as st
from agents.orchestrator import run_sales_meeting_prep

st.set_page_config(page_title='AI Sales Copilot', layout='wide')

st.title('AI Multi-Agent Sales Copilot')
st.markdown('Prepare FMCG sales representatives before customer meetings using AI insights.')

company = st.text_input('Company/Product', 'Cadbury Dairy Milk')
competitor = st.text_input('Competitor', 'Nestle')

if st.button('Generate Intelligence Report'):
    with st.spinner('Running AI agents...'):
        result = run_sales_meeting_prep(company, competitor)

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
