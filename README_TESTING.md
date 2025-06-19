# PyMilvus-PG Testing Guide

本文档介绍如何运行 PyMilvus-PG 项目的测试套件。

## 测试概述

本项目包含以下类型的测试：

- **单元测试** - 测试基础功能，不需要外部服务
- **集成测试** - 测试与 Milvus 和 PostgreSQL 的集成
- **性能测试** - 性能基准测试和压力测试
- **端到端测试** - 完整的工作流程测试

## 快速开始

### 运行本地测试（推荐开始）

运行不需要外部服务的测试：

```bash
# 使用脚本运行
./scripts/run_local_tests.sh

# 或使用 Makefile
make test-quick
```

### 运行所有测试

```bash
make test
```

## 测试分类

### 1. 单元测试

测试基本功能，无需外部依赖：

```bash
# 测试日志配置
pdm run pytest tests/test_logger_config.py -v

# 测试工具类
pdm run pytest tests/test_utils.py -v

# 运行所有单元测试
make test-unit
```

### 2. 集成测试

需要 Milvus 和 PostgreSQL 服务：

```bash
# 设置环境变量
export MILVUS_URI="http://localhost:19530"
export PG_CONN="postgresql://postgres:admin@localhost:5432/test_db"

# 运行集成测试
make test-integration
```

### 3. 性能测试

压力测试和性能基准测试：

```bash
make test-performance
```

## 使用 Docker 运行测试

### 完整的 Docker 环境

使用 Docker Compose 运行完整的测试环境：

```bash
# 启动测试环境
make docker-test

# 停止测试环境
make docker-test-down
```

## 测试配置

### 环境变量

测试使用以下环境变量：

- `MILVUS_URI` - Milvus 服务器地址（默认：http://localhost:19530）
- `PG_CONN` - PostgreSQL 连接字符串（默认：postgresql://postgres:admin@localhost:5432/test_db）

### 测试标记

使用 pytest 标记来运行特定类型的测试：

```bash
# 只运行集成测试
pdm run pytest -m integration

# 只运行性能测试
pdm run pytest -m slow

# 排除性能测试
pdm run pytest -m "not slow"
```

## 代码质量检查

### 代码检查

```bash
# 运行 ruff 检查
make lint

# 自动修复检查问题
make lint-fix
```

### 代码格式化

```bash
# 检查格式
make format-check

# 格式化代码
make format
```

### 类型检查

```bash
make type-check
```

### 测试覆盖率

```bash
make test-coverage
```

## CI/CD

### GitHub Actions

项目配置了以下 GitHub Actions 工作流：

- **test.yml** - 运行测试套件
- **lint.yml** - 代码质量检查
- **release.yml** - 发布流程

### 本地预提交检查

运行所有预提交检查：

```bash
make pre-commit
```

## 故障排除

### 常见问题

1. **Milvus 连接失败**
   ```bash
   # 检查 Milvus 是否运行
   curl http://localhost:9091/healthz
   ```

2. **PostgreSQL 连接失败**
   ```bash
   # 检查 PostgreSQL 是否运行
   pg_isready -h localhost -p 5432 -U postgres
   ```

3. **依赖问题**
   ```bash
   # 重新安装依赖
   pdm install --dev
   ```

### 调试测试

运行单个测试文件：

```bash
pdm run pytest tests/test_milvus_pg_client.py::TestMilvusPGClientBasic::test_client_initialization -v -s
```

查看详细输出：

```bash
pdm run pytest tests/ -v -s --tb=long
```

## 贡献测试

### 添加新测试

1. 在相应的测试文件中添加测试方法
2. 使用适当的测试标记（`@pytest.mark.integration`、`@pytest.mark.slow`）
3. 遵循现有的测试命名约定
4. 确保测试具有适当的清理逻辑

### 测试最佳实践

- 使用描述性的测试名称
- 每个测试只测试一个功能点
- 使用 fixtures 来复用测试设置
- 在测试后进行适当的清理
- 为集成测试添加适当的等待时间

## 性能基准

运行性能测试以获得基准数据：

```bash
# 运行插入性能测试
pdm run pytest tests/test_performance.py::TestPerformanceBenchmarks::test_insert_performance -v -s

# 运行查询性能测试
pdm run pytest tests/test_performance.py::TestPerformanceBenchmarks::test_query_performance -v -s
```

## 总结

- 开发时使用 `make test-quick` 进行快速测试
- 提交前使用 `make pre-commit` 进行完整检查
- CI/CD 会自动运行完整的测试套件
- 使用 Docker 环境进行一致的测试体验 