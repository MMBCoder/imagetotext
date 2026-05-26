from tavily import TavilyClient


def tavily_search(query, tavily_api_key):

    client = TavilyClient(api_key=tavily_api_key)

    response = client.search(
        query=query,
        search_depth='advanced'
    )

    return response
