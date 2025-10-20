1. ~~Explore the response of the API and figure out what kind of questions can the agent answer~~
2. [In progress] Build the agent orchestrator to handle the 3 functions
    a - Handle multi-turn conversations
    b - Chain tools together - I think this will be automatically handled if we include our planning agent mechanism
5. Use a proper vector database instead of in-memory
    - ~~in progress, i am gonna use weaviate~~
    - ~~need to implement query rewriting to extract intent, keywords and phrases, filters for better search results~~
6. Use real government schema data - scrape from official websites, use PDFS etc instead of mock data
7. Implement the market trend analysis tool

