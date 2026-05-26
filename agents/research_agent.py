from tools.tavily_tools import tavily_search


def research_company(company, tavily_api_key):

    query = f'{company} customer reviews market trends sales insights'

    return tavily_search(query, tavily_api_key)
