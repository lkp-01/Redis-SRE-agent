# Outcome Evaluation Contract Design

## 目标

语义 Judge 只能把场景明确声明的必需结论判为阻断性缺项，不能因为参考答案包含更多细节就让高质量回答失败。禁止结论保持单向约束：只有回答实际主张了禁止内容才失败，未主动复述反向警告不算缺项。技术错误继续作为硬失败，但 Judge 必须把纯措辞、可选证据和改进建议留在非阻断字段。

## 设计

场景 YAML 仍以 `required_findings` 和 `forbidden_claims` 为唯一规范来源。构造 Judge 上下文时，为两类条目生成稳定 ID，例如 `required_finding_1` 和 `forbidden_claim_1`。Judge 的 `missing_elements` 与 `violated_forbidden_claims` 只能返回这些 ID；运行时把合法 ID 映射回人类可读描述。旧 Judge 若返回与规范文本完全相同的描述，也继续兼容。任何无法映射到规范条目的“缺项”都会进入 `advisory_missing_elements`，保留反馈但不阻断通过。

参考答案继续保留在测试载荷中供调试，但不再注入 Judge prompt，避免把示例答案误当成穷举 checklist。Judge prompt 明确说明：覆盖可以分散在回答不同段落，不要求逐字匹配或集中成组；禁止项只有在回答明确支持时才算违反；`factual_errors` 只用于会影响诊断或处置的实质错误。

最终通过条件是：分数达到阈值、没有实质事实错误、没有规范内必需项缺失、没有禁止项违规。可选证据遗漏只降低分数或进入 advisory，不再造成零容忍失败。

## 场景数据

FailoverFlapping 的 Prometheus 样本时间改到与 Loki 同一 2026-08-14 时间窗，并让三次 `redis_up` 短暂下降与三次 `switch-master` 对齐。`master_replid2`/`second_repl_offset` 仍作为可选佐证；永久关闭高可用仍只作为禁止结论。

## 验证

- 合法 required ID 缺失必须失败。
- 参考答案独有细节被 Judge 报为缺项时，只记 advisory，不能失败。
- 合法 forbidden ID 被违反时必须失败。
- 低于分数阈值或存在实质事实错误仍必须失败。
- 完整 evaluation 单元测试全部通过。
