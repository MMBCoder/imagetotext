from tools.tavily_tools import tavily_search


def research_company(company):
    query = f'{company} customer reviews market trends sales insights'
    return tavily_search(query)
