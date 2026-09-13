from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator

'''
LangGraph收到前端的请求后，会将前端重名的变量直接将值放入到状态中
比如：
1. 用户问题message——>传入OverallState中的messages
2. 用户选择快速——>对应模型是"qwen3.7-flash-2026-07-15"，根据前端\src\App.tsx中设置的：
    case "low":
    initial_search_query_count = 1;
    max_research_loops = 1;
    那么OverallState中的同名变量也是这个值
'''

# 总体状态
class OverallState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，归约器进行历史消息管理：自动把最新消息添加到消息列表中
    search_query: Annotated[list, operator.add]  # 搜索查询列表
    web_search_result: Annotated[list, operator.add]  # 搜索结果列表
    sources_gathered: Annotated[list, operator.add]  # 搜集的资料来源列表
    '''
    初始搜索查询的关键词数量
    1.token/MCP免费次数有限
    2.搜索模式初始数量不同，如选择快速：初始搜索关键词数量=1，均衡：初始搜索关键词数量=5，专家模式：初始搜索关键词数量=10
    '''
    initial_search_query_count: int
    max_research_loops: int  # 最大研究循环次数：系统会进行反思，最多反思几次
    research_loop_count: int  # 当前的研究次数，如果当前次数=max次数，就停止反思，输出最终报告
    reasoning_model: str  # 当前选择的推理模型名称

''' 节点的临时状态，只在当前节点使用'''
# 反思节点的状态
class ReflectionState(TypedDict):
    is_sufficient: bool  # 当前的研究是否足够，是否要进行反思，如果问题很简单就不用达到最大反思次数才停止
    knowledge_gap: str  # 知识的差距描述，反思前和反思后的差别
    follow_up_queries: Annotated[list, operator.add]  # 为了补足知识的差距，后续要进行哪些查询
    research_loop_count: int  # （总体状态）当前的研究次数，如果当前次数=max次数，就停止反思
    number_of_ran_queries: int  # 已经执行的查询数量
    max_research_loops: int

# 查询节点的状态
class Query(TypedDict):
    query: str  # 把用户问题 拆成 搜索关键字
    rationale: str  # 为什么要选择这个搜索关键字的理由，便于检查拆的关键字合不合理

class QueryGenerationState(TypedDict):
    search_query: list[Query]  # 把拆出来的关键字作为列表封装到 搜索关键字 这个状态中

# 网络搜索
class WebSearchState(TypedDict):
    search_query: str
    id: str

# 网络搜索结果
@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
