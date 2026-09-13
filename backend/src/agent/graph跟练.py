import json

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from IPython.display import Image, display

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.post import Post
from agent.utils import (
    get_research_topic,
    resolve_urls,
)
from agent.base_agent import Agent, JsonAgent, WebSearchAgent
from loguru import logger
load_dotenv()

'''-------------节点名称定义-------------------------'''
# 查询生成节点
GENERATE_SEARCH_NODE = "generate_search"
# web搜索节点
WEB_SEARCH_NODE = "web_search"
# 反思节点
CRITIQUE_NODE = "critique"
# 最终答案生成节点
FINAL_ANSWER_NODE = "final_answer"

def generate_search(state: OverallState, config: Configuration) -> QueryGenerationState:
    '''
    该节点把用户的自然语言请求，拆解出一个个搜索关键词
    0. 传入相关程序配置（前端用户选择了哪个模型）
    1. 拿到用户问题
    2. 大模型去拆解
       2.1：要有提示词
       2.2：要和大模型通信
       2.3：处理大模型返回的文本答案，格式化输出
    3. 把结果往后传递
    '''
    configurable = Configuration.from_runnable_config(config)
    # 如果状态中没拿到 initial_search_query_count，就从初始配置中获取
    if state.get['initial_search_query_count'] is None:
        state.get['initial_search_query_count'] = configurable.number_of_initial_queries
    # 实例化JsonAgent类
    agent = JsonAgent(model_id=configurable.query_generator_model, keys=SearchQueryList)
    # 调用类的方法
    agent.set_step_prompt(query_writer_instructions)  # 传入该节点提前写好的提示词
    result = agent.step(number_queries=state['initial_search_query_count'],
               current_date=get_current_date(),
               research_topic=get_research_topic(state['messages']),
               )
    # 模型返回值就是需要搜索的关键词，更新search_query状态的值
    return { "search_query":result }

def send_to_web_search(state:QueryGenerationState):
    return [
        Send(WEB_SEARCH_NODE, {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]

'''
1.传入关键词进行搜索——MCP服务搜索
2.整合搜索后的内容——大模型整合
'''
def web_search(state:WebSearchState,config:Configuration) -> OverallState:
    '''
     0.传入程序配置
     1.调用阿里云MCP服务，他会返回很多内容
        1.1 搜索结果提炼
        1.2 类似长URL，对内容本身没太大关系的内容进行处理（作为溯源链接）
     2. 把结果往后传
    '''
    configurable = Configuration.from_runnable_config(config)
    # 初始化模型客户端
    web_search = WebSearchAgent()
    # 调用大模型
    response = web_search.step(prompt=state['search_query'],count=10)
    # 检查搜索结果是否为空或None
    if not response or response is None:
        logger.error(f"网络搜索返回结果为空：{state['search_query']}")
        return {
            "sources_gathered":[],
            "search_query":state['search_query'],
            "web_search_result":[f"未找到关于'{state['search_query']}'的搜索结果"],
        }

    # 长短url映射
    long2short_url_mappings = resolve_urls(response, state['id'])
    sources_gathered = [
        {'short_url': long2short_url_mappings[item['url']],'value':item['url'],'label':item['title']} for item in response
    ]
    web_search_result = [
        {'snippet': item['snippet'], 'ttile':item['ttile'],'url':long2short_url_mappings[item['url']]}  for item in response
    ]
    web_search_result = json.dumps(web_search_result, ensure_ascii=False, indent=4)

    '''
    整合搜索结果
    # 创建Agent：使用查询生产模型
    # 设置提示词：加载 web_searcher_instruction 模板
    # 执行总结：传入搜索词、日期、原始结构，让大模型提炼关键信息并标注来源
    # 提取文本：从大模型响应中解析出
    '''
    agent = Agent()
    agent.set_step_prompt(web_searcher_instructions)
    modified_text = agent.step(current_date=get_current_date(),
                        query=state['search_query'],
                        web_search_result=web_search_result,
                        )
    modified_text = Post.extract_pattern(modified_text, pattern='text')

    logger.info(f"搜索标题：{state['search_query']}")
    logger.debug(f"网络搜索结果{modified_text}")
    '''
    虽然返回的是OverallState, 但在return只用写OverallState中需要更新的变量
        sources_gathered:收集的来源列表（含短URL、原始URL、标题）
        search_query:将当前查询词包装为列表
        web_search_results:将LLM总结的文本包装为列表
    '''

    return {
        'sources_gathered':sources_gathered,
        'web_search_result':[modified_text],
        'search_query':state['search_query'],
    }

def critique(state: OverallState, config:Configuration) -> ReflectionState:
    '''
    识别只是差距并生成潜在后续查询的节点

    分析当前摘要以识别需要进一步研究的领域，并生成潜在的后续查询
    使用结构化输出来提取JSON格式的后续查询。
    Args:
        state (OverallState):包含运行摘要和研究主题的当前图状态
        config:可运行配置，包括LLM提供商设置
    Returns:
          包含状态更新的字典，包括search_query键，包含生成的后续查询
    '''
    logger.info(f'反思分析识别知识差距并生成潜在后续查询的节点工作')
    configurable = Configuration.from_runnable_config(config)
    # 增加研究循环计数并获取推理模型
    state['research_loop_count'] = state.get('research_loop_count', 0) + 1
    reasoning_model = state.get('research_model',configurable.reflection_model)
    logger.info(f'critique反思节点模型：{reasoning_model}')

    # 格式化提示，设置大模型返回格式为Reflection
    agent = JsonAgent(model_id=reasoning_model,keys=Reflection)
    agent.set_step_prompt(reflection_instructions)
    result = agent.step(research_topic=get_research_topic(state['messages']),
                        current_date=get_current_date(),
                        number_queries=state['max_research_loops '],
                        summaries="\n\n---\n\n".join(state["web_search_result"]))

    logger.info(f'反思分析：{result}')
    return {
        'is_sufficient':result.is_sufficient,
        'knowledge_gap':result.knowledge_gap,
        'follow_up_queries':result.follow_up_queries,
        'research_loop_count':state['research_loop_count'],
        'number_of_ran_queries':state['search_query'],  # 已经执行的搜索有几个
        'max_research_loops':result.max_research_loops,
    }

def route_evaluate(state: ReflectionState, config:Configuration) -> OverallState:
    '''
    确定反思节点后下一步的路由函数
    通过'is_sufficient'信息是否足够参数 或 'max_research_loops'最大研究循环次数 来控制循环次数

    Args:
        state:包含研究循环计数的当前图状态
        config:可运行配置，用于获取max_research_loops
    Return:
         字符串，指示下一个要访问的节点
    '''
    logger.info('准备评估当前研究......')
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get('max_research_loops')
        if state.get('max_research_loops') is not None
        else configurable.max_research_loops
    )

    logger.info(state)
    logger.info(f'最大研究循环次数：{max_research_loops}')
    logger.info(f'当前已研究次数：{state["research_loop_count"]}')
    if state['is_sufficient'] or state['research_loop_count '] >= max_research_loops:
        return  FINAL_ANSWER_NODE
    else:  # 把反思节点新生成的查询关键词分发到搜索节点中
        return [
            Send(WEB_SEARCH_NODE,{'search_query':follow_up_query,'id':state['number_of_ran_queries'] + int(idx)})
            for idx,follow_up_query in enumerate(state['follow_up_queries'])
        ]

def final_answer(state: OverallState, config:Configuration) -> OverallState:
    """
    最终确定research摘要的节点

    通过去重和格式化源，然后将它们与运行摘要结合，
    创建结构良好的研究报告，包含适当的引用。

    Args:
        state: 包含运行摘要和收集源的当前图状态

    Returns:
        包含状态更新的字典，包括running_summary键，包含格式化的最终摘要和源
    """
    logger.info('最终答案准备生成......')
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get('reasoning_model') or configurable.answer_model
    logger.info(f'final_answer最终答案节点模型：{reasoning_model}')

    # 格式化提示
    agent = Agent(model_id=reasoning_model)
    agent.set_step_prompt(answer_instructions)
    content = agent.step(current_date=get_current_date(),
                        research_topic=get_research_topic(state['messages']),
                        summaries="\n\n---\n\n".join(state["web_search_result"]))

    # 资料来源链接：用原始URL替换短URL，并将所有使用的URL添加到source_gathered
    unique_source = []
    for source in state['sources_gathered']:  # 遍历列表中的每一个来源，可能有的搜到了，但是最终生成报告没有用
        if source['short_url'] in content:  # 检查：这个短链接的字符串，是否出现在大模型刚生成的 content（报告原文）
            content = content.replace(source["short_url"], source["value"])
            unique_source.append(source)

    logger.info(f"最终确定答案：{content}")
    return {
        "messages": [AIMessage(content=content)],
        "sources_gathered": unique_source,
    }

# 构建一个空白图，传入总体状态+配置
builder = StateGraph(OverallState, config_shchema=Configuration)

# 在空白图中加入节点
builder.add_node(GENERATE_SEARCH_NODE, generate_search)
builder.add_node(WEB_SEARCH_NODE, web_search)
builder.add_node(CRITIQUE_NODE, critique)
builder.add_node(FINAL_ANSWER_NODE, final_answer)

# 添加边
builder.add_edge(START, GENERATE_SEARCH_NODE)
'''
将用户的问题拆解为查询关键词，拆成多个关键词，可能会生成多个查询请求——>send分发,同时进行查询
'''
builder.add_conditional_edges(
    GENERATE_SEARCH_NODE,send_to_web_search,[WEB_SEARCH_NODE]
)
builder.add_edge(WEB_SEARCH_NODE, CRITIQUE_NODE)
'''
把搜索结果反思后进行处理：如果有问题、搜索的不够：就重新搜索，如果没问题就进行最后的输出
'''
builder.add_conditional_edges(
    CRITIQUE_NODE, route_evaluate,[WEB_SEARCH_NODE,FINAL_ANSWER_NODE])
builder.add_edges(FINAL_ANSWER_NODE, END)

# 编译图
builder.compile(name="pro_research_agent")