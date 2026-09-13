# DeepResearch Quick Start

当前项目展示了如何使用Langgraph搭建一个DeepResearch的应用
<img src="./app.png" title="Use Langgraph to build an DeepResearch" alt="如何使用Langgraph搭建一个DeepResearch的应用" width="90%">

## 项目结构
当前项目目录分为以下两个结构
-   `frontend/`: 项目前端
-   `backend/`: 包含了核心的后端逻辑，所有的Agent体系的后端逻辑都在当前目录下

## Quick Start
**1. 前期准备:**
- Node.js and npm (or yarn/pnpm)
- Python 3.11+
- miniconda或anaconda
- API Key，可以从[百炼](https://bailian.console.aliyun.com/)官网注册登录获取

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
pip install .

pip install langgraph>=0.2.6
pip install langchain>=0.3.19
pip install openai
pip install python-dotenv>=1.0.1
pip install langgraph-sdk>=0.1.57
pip install langgraph-cli
pip install langgraph-api
pip install fastapi
```

**Frontend:**

```bash
cd frontend
npm install
```

**3. Run Development Servers:**
配置好相关的APIKey后，运行以下命令启动后端服务
```bash
run_backend.bat
.\run_backend.bat
```

运行以下命令启动前端服务
```bash
run_fontend.bat
.\run_fontend.bat
```
