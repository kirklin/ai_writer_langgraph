import functools, operator, os
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage
)

# 设置环境变量
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
os.environ["LANGCHAIN_PROJECT"] = "LangGraph"

# 初始化语言模型
llm = ChatOpenAI(model="gpt-4o",
                 api_key=OPENAI_API_KEY,
                 base_url=OPENAI_BASE_URL)


# 定义生成大纲工具
@tool("generate_outline")
def generate_outline(topic: str) -> str:
    """根据给定的主题生成大纲。"""
    chat = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    messages = [
        SystemMessage(
            content="你是一个大纲生成器。"
        ),
        HumanMessage(
            content=f"请为以下主题生成大纲：{topic}"
        ),
    ]
    response = chat(messages)
    return response.content


# 定义写文章工具
@tool("write_article")
def write_article(outline: str) -> str:
    """根据给定的大纲写一篇文章。"""
    chat = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    messages = [
        SystemMessage(
            content="你是一个文章写手。"
        ),
        HumanMessage(
            content=f"请根据以下大纲写一篇文章：{outline}"
        ),
    ]
    response = chat(messages)
    return response.content


# 定义Agent状态数据类型
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# 创建Agent执行器
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


# 定义Agent节点
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# 设定成员和系统提示
members = ["Outline_Generator", "Article_Writer"]
system_prompt = (
    "你是一个主管，负责管理以下工作者之间的对话：{members}。"
    "根据以下用户请求，选择下一个执行任务的工作者。每个工作者会执行一个任务并回复其结果和状态。"
    "任务完成后，请回复‘完成’。"
)

# 设置选项和函数定义
options = ["完成"] + members
function_def = {
    "name": "route",
    "description": "选择下一个Agent。",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

# 创建Supervisor链
supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "根据以上对话，下一步应该由谁执行？"
            " 还是我们应该‘完成’？请选择其中一个：{options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
        supervisor_prompt_template
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
)

# 创建大纲生成Agent
outline_generator_agent = create_agent(llm, [generate_outline], "你是一个大纲生成器。")
outline_generator_node = functools.partial(agent_node, agent=outline_generator_agent, name="Outline_Generator")

# 创建文章写作Agent
article_writer_agent = create_agent(llm, [write_article], "你是一个文章写手。")
article_writer_node = functools.partial(agent_node, agent=article_writer_agent, name="Article_Writer")

# 创建工作流图
workflow = StateGraph(AgentState)
workflow.add_node("Outline_Generator", outline_generator_node)
workflow.add_node("Article_Writer", article_writer_node)
workflow.add_node("supervisor", supervisor_chain)

# 为每个成员添加边
for member in members:
    workflow.add_edge(member, "supervisor")

# 定义条件映射
conditional_map = {k: k for k in members}
conditional_map["完成"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# 设置入口点
workflow.set_entry_point("supervisor")

# 编译工作流图
compiled_graph = workflow.compile()

# 执行工作流图
for state in compiled_graph.stream(
        {
            "messages": [
                HumanMessage(content="请为‘人工智能的未来’生成一个大纲，并根据大纲写一篇文章")
            ]
        }
):
    if "__end__" not in state:
        print(state)
        print("----")
