import os
from tavily import TavilyClient

client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))


def tavily_search(query):
    response = client.search(query=query, search_depth='advanced')
    return response
