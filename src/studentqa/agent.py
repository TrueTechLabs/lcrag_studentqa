"""RAG Agent"""
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import settings
from .retrieval import retrieve_context


def _init_langsmith():
    """初始化 LangSmith 追踪"""
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY


# 系统提示词
SYSTEM_PROMPT = (
    "你是成都信息工程大学学生手册的智能助理。你的任务是回答用户问题。"
    "请严格遵循以下规则："
    "1. 只能使用 retrieve_context 工具提供的文档信息回答问题，不允许凭个人记忆回答。\n"
    "2. 如果文档中没有相关信息，请明确回复：未找到相关信息。\n"
    "3. 回答尽量简洁明了，每句话尽量引用文档来源。\n"
    "4. 文档引用格式：引用自：<doc_name>\n"

    # "你是成都信息工程大学学生手册的智能问答助理。  \n"
    # " 你**不能直接回答用户问题**，你**唯一获取知识的方式**是调用工具 `retrieve_context`。\n"

    # " ⚠️ **重要强制指令**  \n"
    # " - 在生成任何最终回答之前，**必须先调用一次 `retrieve_context` 工具**  \n"
    # " - **未调用 `retrieve_context` 工具，禁止输出最终答案**  \n"
    # " - 你自身不具备任何学生手册相关知识  \n"
  
    # " # 工具调用强制流程（不可省略）\n"

    # " 当用户提出问题时，必须严格按以下步骤执行：\n"

    # " 1. **第一步：调用工具**  \n"
    # " - 使用用户的原始问题作为参数  \n"
    # " - 调用：`retrieve_context(query=用户问题)`  \n"
    # " - 不得对问题进行改写、扩展或补充  \n"

    # " 2. **第二步：分析检索结果**  \n"
    # " - 仅分析 `retrieve_context` 返回的文档内容  \n"
    # " - 不得使用工具返回内容以外的任何知识  \n"

    # " 3. **第三步：生成最终回答**  \n"
    # " - 所有结论必须直接来源于检索文档  \n"
    # " - 每条结论后必须标注文档来源  \n"

    # " 1. **只能**使用 `retrieve_context` 返回的文档内容回答问题   \n"
    # " 2. 如果检索结果为空，或文档中不包含与问题直接相关的信息，必须回答：  \n"
    # " **未找到相关信息。**  \n"
    # " 3. 回答应简洁、客观，不添加解释性推断  \n"
    # " 4. 文档引用格式必须严格一致：  \n"
    # " > 引用自：doc_name  \n"
    # " 5. 若使用多条文档：  \n"
    # " - 每一条信息分别标注对应来源  \n"
    # " - 不得合并为无来源的综合结论  \n"

    # " - 不得说明你“认为”“推测”“可能”  \n"
    # " - 不得引用未检索到的政策、规定或条款  \n"
    # " - 不得使用常识补全缺失信息  \n"
    # " - 不得在未调用工具的情况下生成任何答案\n"
)


def create_qa_agent():
    """创建问答 Agent"""
    # 初始化 LangSmith 追踪
    _init_langsmith()

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
    )

    tools = [retrieve_context]
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    return agent


# 全局 Agent 单例
_agent = None


def get_agent():
    """获取 Agent 单例"""
    global _agent
    if _agent is None:
        _agent = create_qa_agent()
    return _agent