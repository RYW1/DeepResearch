:: set APP_TOKEN=你的LLM API Key
:: set LLM_BASE_URL=你的LLM服务URL

:: set MCP_APP_ID=你的MCP工具ID
:: 明确禁用 LangSmith/LangChain 追踪
@REM set LANGCHAIN_TRACING_V2=false
@REM set LANGCHAIN_ENDPOINT=
@REM set LANGCHAIN_API_KEY=
@REM set LANGCHAIN_PROJECT=
:: LANGSMITH_API_KEY=你的LangSmith API

:: 设置UTF-8编码以解决Windows中文系统下的编码问题
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd backend

pip install -e .
langgraph dev --no-browser --allow-blocking
::langgraph serve