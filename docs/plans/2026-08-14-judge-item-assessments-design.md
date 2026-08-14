# Judge Item Assessments Design

## 目标

修复语义 Judge 在文字反馈声称“满足所有必需项”的同时，把全部 required ID 放进 `missing_elements` 的自相矛盾行为。运行时不再信任一个无证据的缺失列表，而是要求 Judge 对每个必需项和禁止项逐项给出状态与回答原文证据，再由确定性代码推导阻断结果。

## 协议

Judge 必须返回 `required_element_results`。每个目录 ID 恰好出现一次，状态只能是 `present`、`partial` 或 `missing`。`present` 和 `partial` 必须附带非空 `evidence`；对于跨段落的组合要求，允许使用简短、可审计的归纳证据，避免强迫模型伪造一个不存在的连续原文片段。`missing` 的 evidence 必须为空。只有 `missing` 进入现有 `missing_elements` 硬失败字段，`partial` 进入新增的 `partial_elements`，用于扣分和反馈但不单独阻断高于阈值的回答。

禁止项使用同样的 `forbidden_claim_results`，状态只能是 `absent` 或 `violated`。`violated` 必须提供回答原文证据，只有它会进入 `violated_forbidden_claims`。这消除了“没有主动反对禁止项也算缺失”的歧义。

## 一致性与恢复

归一化层验证目录覆盖、重复/未知 ID、状态值和证据存在性；对真正会阻断 Agent 的 forbidden violation 继续要求证据是回答原文。首次 Judge 结果违反协议时，使用单独的 contract-repair prompt 重新评估一次，而不是静默猜测。即使首轮合同有效，只要它产生 missing、forbidden violation 或 factual error 这类阻断结论，也必须进行一次独立确认；只有确认结果仍支持阻断才判失败。第二次结果无效时，返回 `judge_valid=false` 和具体 `validation_errors`。运行时将 `judge_passed` 与顶层 `passed` 设为 `None`，CLI 输出 `passed=invalid` 并报 `Outcome evaluation invalid`，明确这是评测基础设施错误，不是 Agent 未通过。

原有 `missing_elements`、`violated_forbidden_claims` 和 advisory CLI 输出继续保留，以减少调用方迁移成本。Judge JSON 示例中的阻断数组改为空或完全移除，避免再次诱导模型复制 placeholder ID。

## 测试

- 四项均 `present` 时，即使反馈包含 advisory，也允许通过。
- `partial` 不进入 blocking missing。
- `missing` 由逐项状态确定并映射为规范描述。
- evidence 不是回答子串、ID 重复或目录不完整时触发一次重判。
- 重判仍无效时结果为 invalid，而非 Agent fail。
- forbidden `violated` 有原文证据时继续硬失败。
