# 仓库规则

## 分支创建

仅以下 GitHub 账号允许在主仓库创建分支：

- `juyangokok`
- `weinachuan`
- `weiwei-612`（Sun Weiwei）
- `woey`
- `zhangshuolei-hfut`
- `chen-linxin`

其他成员请 fork 本仓库到个人仓库，并从个人仓库向主仓库提交 Pull Request。

`.github/workflows/repository-rules.yml` 中的 `branch-creation-guard` 会在主仓库出现未授权新分支时自动删除该分支，并提示提交者走 fork + PR 流程。

## 合入检视

PR 合入前需要上述维护账号中的 2 个账号在当前 head commit 上检视通过。`Repository Rules / required-reviewers` workflow 会检查有效审批数，并排除 PR 提交人自己的审批。

`.github/CODEOWNERS` 将全仓默认归属到维护账号组，便于 GitHub 自动请求检视。仓库管理员应将 `Repository Rules / required-reviewers` 配置为 `main` 分支必需状态检查。

## 强行合入

`weinachuan` 的强行合入权限需要在 GitHub 分支保护中配置 PR bypass allowance。仓库管理员可使用以下脚本应用 `main` 分支保护：

```sh
GITHUB_TOKEN=<admin-token> scripts/github/apply_branch_protection.sh main
```

该 token 需要具备仓库 administration 写权限。脚本会要求 `Repository Rules / required-reviewers` 通过，并配置 2 个 approval、CODEOWNERS review、stale review dismiss 和 `weinachuan` bypass。
