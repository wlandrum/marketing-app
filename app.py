import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool, YoutubeChannelSearchTool
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
# Force Chroma to use in-memory mode (avoiding sqlite3)
os.environ["PERSIST_DIRECTORY"] = "none"


st.set_page_config(page_title="Enhanced Music Marketing Planner", layout="centered")

st.title("🎶 Enhanced Music Marketing Planner")

single_name = st.text_input("🎵 Name of your new single/album:")
style_tone = st.text_area("✨ Describe your desired style/tone (casual, edgy, etc.):")
youtube_channel = st.text_input(
    "📺 Enter YouTube Channel or Artist Name (optional, recommended):",
    help="Example: Ed Sheeran, Billie Eilish, etc."
)

if st.button("Generate Marketing Plan"):
    if not single_name or not style_tone:
        st.warning("Please provide both the single/album name and desired style.")
    else:
        st.toast("Starting marketing plan generation... 🚀")

        llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.5
        )

        search_tool = SerperDevTool(api_key=serper_api_key)

        youtube_channel_tool = YoutubeChannelSearchTool(
            config=dict(
                llm=dict(
                    provider="openai",
                    config=dict(
                        model="gpt-4-turbo",
                        temperature=0.5
                    ),
                ),
                embedder=dict(
                    provider="openai",
                    config=dict(
                        model="text-embedding-3-small",
                    ),
                ),
            )
        )

        strategist = Agent(
            role="Music Marketing Strategist",
            goal="Create an actionable marketing strategy using current trends",
            backstory="Expert music marketer who uses real-time insights to create relevant marketing strategies.",
            llm=llm,
            allow_delegation=False,
            tools=[search_tool]
        )

        creator = Agent(
            role="Content Creator",
            goal="Develop engaging promotional content leveraging YouTube channel insights",
            backstory="Creative professional crafting impactful promotional content by analyzing YouTube trends.",
            llm=llm,
            allow_delegation=False,
            tools=[youtube_channel_tool]
        )

        scheduler = Agent(
            role="Social Scheduler",
            goal="Suggest ideal timing for social media content",
            backstory="Analytics-driven expert optimizing posting schedules.",
            llm=llm,
            allow_delegation=False
        )

        reviewer = Agent(
            role="Reviewer",
            goal="Ensure content quality and brand alignment",
            backstory="Experienced editor ensuring content quality and consistency.",
            llm=llm,
            allow_delegation=False
        )

        with st.status("✨ Generating your personalized marketing plan...", expanded=True) as status:
            st.write("🧑‍💻 **Creating Marketing Strategy...**")
            strategy_task = Task(
                description=(
                    f"Create a concise marketing strategy for '{single_name}' with style '{style_tone}'. "
                    "Use current trends in the music industry, popular hashtags, and competitor insights. "
                    "Include target audience, recommended platforms, and promotional angles."
                ),
                expected_output="Brief, actionable marketing plan based on real-time insights.",
                agent=strategist
            )

            st.write("🎨 **Generating Content Ideas...**")
            if youtube_channel.strip() != "":
                content_description = (
                    f"Generate engaging content ideas for Instagram posts, TikTok videos, and promotional emails. "
                    f"Analyze the YouTube channel '{youtube_channel}' to leverage successful promotional strategies and video trends."
                )
            else:
                content_description = (
                    "Generate engaging content ideas for Instagram posts, TikTok videos, and promotional emails. "
                    "Use general trending video insights in the music industry."
                )

            content_task = Task(
                description=content_description,
                expected_output="Clear, relevant, and engaging content ideas based on provided YouTube insights.",
                agent=creator,
                context=[strategy_task]
            )

            st.write("📅 **Determining Optimal Posting Schedule...**")
            schedule_task = Task(
                description="Provide optimal posting schedule recommendations for social media. Always output the schedule in a table",
                expected_output="Suggested posting schedule (days and times). Always output the schedule into a table.",
                agent=scheduler,
                context=[content_task]
            )

            st.write("🔍 **Reviewing and Refining Output...**")
            review_task = Task(
                description="Review and refine all content for quality and alignment.",
                expected_output="Polished final marketing strategy, content, and schedule.",
                agent=reviewer,
                context=[strategy_task, content_task, schedule_task]
            )

            crew = Crew(
                agents=[strategist, creator, scheduler, reviewer],
                tasks=[strategy_task, content_task, schedule_task, review_task]
            )

            marketing_plan = crew.kickoff()

            status.update(label="✅ Marketing plan ready!", state="complete", expanded=False)

        st.success("Marketing plan generated successfully!")
        st.subheader("📌 Your Marketing Plan:")
        st.markdown(marketing_plan, unsafe_allow_html=True)
