"""DeepResearch多智能体
由三个子智能体组成：
1. 计划阶段（图内，包含人机交互）
2. 研究智能体（子图）— 查询 → 搜索 → 评估循环
3. 写作智能体（子图）— 提纲 → 草稿 → 引用和润色
原有的单体图已重构，每个阶段都成为一个自包含、可独立测试的子图。
"""

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage
from IPython.display import Image, display
from loguru import logger

from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    plan_instructions,
    plan_reflection_instructions,
)
from agent.post import Post
from agent.state import OverallState
from agent.tools_and_schemas import PlanReflection
from agent.utils import (
    get_last_user_response,
    get_research_topic,
)
from agent.base_agent import Agent, JsonAgent
from agent.sub_agents import research_agent_graph, writer_agent_graph

from backend.src.agent.base_agent import Agent
from backend.src.agent.configuration import Configuration
from backend.src.agent.graph import builder, evaluate_plan

load_dotenv()

GENERATE_PLAN_NODE = "generate_plan"
RESEARCH_AGENT_NODE = "research"
WRITER_AGENT_NODE = "write"

def generate_plan(state: OverallState,config: Configuration)->dict:
    if state.get('plan_status','unconfirmed') != 'unconfirmed':
        return {}

    configurable = Configuration.from_runnable_config(config)
    agent = JsonAgent(model_id=configurable.query_generator_model)
    agent.set_step_prompt(plan_instructions)
    response = agent.step(
        current_date=get_current_date(),
        research_topic=get_research_topic(
            state["messages"],
            [m.content for m in state.get("plan_messages", [])],
        ),
        research_proposal=state.get("plan", ""),
    )
    response = Post.extract_pattern(response, pattern="markdown")
    logger.info(f"[MainGraph] 生成的计划 ({len(response)} 字)")

    return {
        'messages': [AIMessage(content=response)],
        'plan': response,
        'plan_status': 'unconfirmed',
        'plan_messages': [AIMessage(content=response)],
    }

builder = StateGraph(state=OverallState,config=Configuration)

builder.add_node(GENERATE_PLAN_NODE,generate_plan)
builder.add_node('replan', lambda state,config:state)
builder.add_node('awaiting_plan_confirmation', lambda state,config:state)

# -- 子图节点，把子图作为主图的一个节点--
builder.add_node(RESEARCH_AGENT_NODE,research_agent_graph)
builder.add_node(WRITER_AGENT_NODE,writer_agent_graph)

# 添加边
builder.add_edge(START,GENERATE_PLAN_NODE)
builder.add_conditional_edges(
    GENERATE_PLAN_NODE,
    evaluate_plan,
    [RESEARCH_AGENT_NODE,'replan','awaiting_plan_confirmation']
)
builder.add_edge('replan',GENERATE_PLAN_NODE)
builder.add_edge(RESEARCH_AGENT_NODE, WRITER_AGENT_NODE)
builder.add_edge(WRITER_AGENT_NODE, END)

graph = builder.compile(name=pro_research_agent)