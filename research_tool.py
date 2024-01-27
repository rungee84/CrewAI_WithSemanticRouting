# research_tool.py
from crewai import Agent, Task, Crew
from langchain.tools import tool, DuckDuckGoSearchRun
from nba_tools import NBATools
from langchain.llms import openai
from semantic_router import RouteLayer, Route
from semantic_router.encoders import FastEmbedEncoder
from browser_tools import BrowserTools

# Initialize the tools and client
search_tool = DuckDuckGoSearchRun()
nba_tools_instance = NBATools()
browser_tools_instance = BrowserTools()
client = openai.OpenAI(base_url="http://127.0.0.1:5000/v1", openai_api_key="test")

# Define the NBA Stats Researcher Agent
class NBAStatsResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role='NBA Stats Researcher',
            goal="Conduct in-depth research on NBA statistics. Only ever use verified, up to date information found from your tools.",
            backstory="""
               Skilled in leveraging both search tools and the 'fetch_nba_stats' tool for comprehensive NBA statistical analysis. Here's how I approach complex queries:

                - For broad or team-based statistics, or to find a roster, use the 'fetch_nba_stats' tool with the team abbreviation. This provides detailed per-game player stats and overall team stats from basketball-reference.com.
                - If the query is about a specific player, modify the search query to include the player's name, ensuring targeted results.
                - Use the DuckDuckGo search tool to gather general statistics and trends, or for additional context beyond the scope of the 'fetch_nba_stats' tool.
                - Employ the custom Google search function for nuanced queries or to gain different perspectives on the statistics.
                - Combine insights from the 'fetch_nba_stats' tool and the search tools to compile a detailed and accurate report on NBA statistics, ensuring a comprehensive understanding of players' and teams' performances.
                - If you are asked for a roster, you must return a list of key players, the person asking cannot view the roster themselves.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer, nba_tools_instance.fetch_nba_stats],
            llm=client
        )

# Define the NBA Injury Researcher Agent
class NBAInjuryResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role='NBA Injury Researcher',
            goal="Quickly find the latest injury info for NBA players. Only ever use verified, up to date information found from your tools.",
            backstory="""
              Expert at finding the latest and most detailed injury information on specific NBA players or teams. Here's how I tackle these specific queries:
              - Use the fetch_nba_injuries tool to find the latest injury information on a player or team.
              - Use both the DuckDuckGo and custom Google search tools to find the latest injury information on a player or team.
              - Examine the fetched data for key details such as injury nature, expected recovery time, and recent updates.
              - If more nuanced information is needed, or to cross-verify, use the DuckDuckGo and custom Google search tools to search for additional details or less commonly reported information.
              - Synthesize information from all sources to provide a comprehensive and current report on the injury status of the player or team, with attention to the latest updates and expert opinions on recovery and potential game participation.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer, nba_tools_instance.fetch_nba_injuries],
            llm=client
        )

# Define the NBA General Researcher Agent
class NBAGeneralResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role='NBA General Researcher',
            goal="Conduct focused and efficient research on specific NBA topics. Only ever use verified, up to date information found from your tools.",
            backstory="""
               Focused on analyzing NBA betting market movements using search tools. My method involves:

                Use DuckDuckGo to search for market trends, like 'NBA betting odds changes 2024'.
                Review the search summaries for patterns or significant shifts in the betting landscape.
                Utilize the Google search function for specific queries, such as 'impact of player X injury on NBA betting odds'. This can uncover detailed insights.
                Integrate data from both tools to provide a well-rounded analysis of the betting market."
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer],
            llm=client
        )


class MarketAnalystAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Betting Market Analyst',
            goal="""Analyze NBA betting market movements and trends to identify profitable betting opportunities. Only ever use verified, up to date information found from your tools.""",
            backstory="""
              Focused on analyzing NBA betting market movements using search tools. My method involves:

                Use DuckDuckGo to search for market trends, like 'NBA betting odds changes 2024'.
                Review the search summaries for patterns or significant shifts in the betting landscape.
                Utilize the Google search function for specific queries, such as 'impact of player X injury on NBA betting odds'. This can uncover detailed insights.
                Integrate data from both tools to provide a well-rounded analysis of the betting market."
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer],
            llm=client
        )

class BettingAdvisorAgent(Agent):
    def __init__(self):
        super().__init__(
            role='NBA Betting Advisor',
            goal="""Provide well-reasoned betting advice combining insights from various analyses. Only ever use verified, up to date information found from your tools.""",
            backstory="""
               Expert in providing betting advice by synthesizing search results. My approach is:

                Search broad betting topics using DuckDuckGo, like 'NBA betting strategies 2024'.
                Analyze summaries to understand general advice and trends.
                For specific scenarios or complex questions, turn to the custom Google search. It often yields more targeted summaries.
                Blend insights from both searches to formulate sound, well-informed betting advice."
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer],
            llm=client
        )


class ExpertOpinionAnalystAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Expert Opinion Analyst',
            goal="""Provide insights based on expert NBA opinions and predictions. Use your tools to gather fresh information.""",
            backstory="""
                Skilled in gathering expert NBA opinions. To tackle challenging questions, I:

                Begin with DuckDuckGo for a general search on expert predictions, like 'NBA expert predictions 2024'.
                Review the result summaries for key opinions and consensus.
                Use the Google search tool for more specific queries or to gain different perspectives.
                Combine the information from both searches to present a comprehensive view of expert opinions."
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer],
            llm=client
        )



class NBATeamPerformanceAnalystAgent(Agent):
    def __init__(self):
        super().__init__(
            role='NBA Team Performance Analyst',
            goal="""Provide detailed analysis on the recent performance of specific NBA teams. Only ever use verified, up to date information found from your tools.""",
            backstory="""
                Focused on delivering detailed performance analysis of NBA teams. My steps are:

                Use DuckDuckGo for initial searches on team performance, like 'Los Angeles Lakers performance 2024'.
                Scan the summaries for recent performance data and trends.
                For more detailed insights, especially on specific players or matches, use the Google search.
                Integrate the data from both searches to provide a thorough analysis of team performance."
            """,
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, browser_tools_instance.google_search_for_answer],
            llm=client
        )

# Define routes for each research area
stats_route = Route(name="stats", utterances=[
    "team statistics", 
    "individual player stats",
    "season averages for players",
    "NBA player scoring leaders",
    "NBA rebounding statistics"
    "roster",
    "team roster",
    "rosters",
    "current NBA standings",
    "current roster of"
    "Research the current rosters for the  Provide a list of their starters."
    "update me on the current roster of the"

])


injury_route = Route(name="injuries", utterances=[
    "player injury updates", 
    "team injury report",
    "injury status of [player name]",
    "[team name] injury list",
    "update on [player name]'s recovery"
])

general_route = Route(name="general", utterances=[
    "season highlights", 
    "game recaps", 
    "upcoming NBA matchups",
    "notable NBA news",
    "recent trends in NBA"
])


market_route = Route(name="market", utterances=[
    "betting line movements", 
    "NBA betting odds", 
    "betting trends in NBA"
])

advisor_route = Route(name="advisor", utterances=[
    "betting advice for NBA games", 
    "NBA betting strategy", 
    "sports betting tips for NBA"
])

expert_route = Route(name="expert", utterances=[
    "expert NBA predictions", 
    "NBA game analysis", 
    "NBA match forecasts"
])



team_performance_route = Route(
    name="team_performance",
    utterances=[
        "team recent performance in NBA", 
        "team winning streaks in NBA", 
        "detailed team game analysis",
        "offensive and defensive ratings of [team name]",
        "Player performance statistics of [team name] in recent games"
    ]
)

# Initialize route layer
encoder = FastEmbedEncoder()
route_layer = RouteLayer(encoder=encoder, routes=[injury_route, general_route, market_route,  expert_route,  team_performance_route, advisor_route, stats_route])

agent_route_map = {
    "stats": {
        "agent_class": NBAStatsResearcherAgent,
        "task_description": "Research the following topic and presenting findings in a few sentences. Approach each topic freshly, ensuring the information is current and succinctly delivered."
    },
    "injuries": {
        "agent_class": NBAInjuryResearcherAgent,
        "task_description": """
            Find and report the latest injury info on NBA players. Keep it simple:
            
            - Who's hurt? What's the injury?
            - How long are they likely out for?
            - Will they play in the next big game?
            - Make sure the info is fresh and real.
            - Keep it short and to the point.
            """
    },
    "general": {
        "agent_class": NBAGeneralResearcherAgent,
        "task_description": "Research the following topic and presenting findings in a few sentences. Approach each topic freshly, ensuring the information is current and succinctly delivered."
    },
    "team_performance": {
        "agent_class": NBATeamPerformanceAnalystAgent,
        "task_description": "Research the following topic and presenting findings in a few sentences. Approach each topic freshly, ensuring the information is current and succinctly delivered."
    },
    "market": {
        "agent_class": MarketAnalystAgent,
        "task_description": "Research the following topic and presenting findings in a few sentences. Approach each topic freshly, ensuring the information is current and succinctly delivered.."
    },
    "advisor": {
        "agent_class": BettingAdvisorAgent,
        "task_description": "Research the following topic and presenting findings in a few sentences. Approach each topic freshly, ensuring the information is current and succinctly delivered."
    },
    "expert": {
        "agent_class": ExpertOpinionAnalystAgent,
        "task_description": "Research the following topic and presenting findings in a few sentences. Approach each topic freshly, ensuring the information is current and succinctly delivered."
    }
}
import datetime

@tool("Natural Language Research Tool")
def natural_language_research(request):
    """
    Conducts research based on a natural language request by routing the request to the appropriate NBA-focused agent and task. 
    This tool utilizes semantic routing to determine the most relevant NBA research area (stats, injuries, general) for a given request and 
    forms a crew to execute the task, returning the result of the research.

    Args:
        request (str): A natural language description of the NBA research topic.

    Returns:
        str: The results of the NBA research, as compiled by the selected agent.
    """



    
    # Determine which route to use
    route_choice = route_layer(request).name
    print(f"Route choice: {route_choice}")

    # Lookup the agent and task for the chosen route
    agent_info = agent_route_map[route_choice]
    agent_instance = agent_info["agent_class"]()

   
     # Ensure the task description includes the directive to use tools for fresh information
    task_description = f"{agent_info['task_description']}: {request}. Remember to use your available tools to gather new information, and base your analysis on current and factual data.  Keep your answers concise and clear while including all the facts. You are researching for an upcoming game that will happen in the future."
    #get today's date
    today = datetime.datetime.now()
    #add today's date to the task description
    task_description += f" Today's date is {today.strftime('%Y-%m-%d')}"

    # Create the task and form the crew
    task = Task(description=task_description, agent=agent_instance)
    crew = Crew(agents=[agent_instance], tasks=[task])

    # Execute the task and return the result
    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    request = "Who is leading in NBA player stats this season?"
    result = natural_language_research(request)
    print(result)
