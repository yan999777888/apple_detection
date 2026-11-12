# 部分验证字段作废说明

本次 `results.json` 的 `validation.AMP_FP16` 不能作为FP16精度证据。

原因：外层torch.autocast被Ultralytics BaseValidator在独立验证模式下的禁用上下文覆盖，实际卷积仍运行FP32。FP32与所谓FP16的官方AP逐位相同触发了源码核查，已定位 `ultralytics/engine/validator.py` 内部上下文。

原始记录保留，不覆盖或删除。该问题不影响本次FP32官方验证，也不影响通过直接model.predict_source/predict_both执行的两种部署路径固定工作点统计及测速；后者的FP16确实产生1个额外FP。

修复：在TaskView.forward内部进入请求的autocast，并通过真实Conv2d输出dtype hook断言FP32为torch.float32、AMP_FP16为torch.float16。独立复核输出在 `../strong_defect_amp_20261112_v2`，不能继续引用v1的AMP官方AP为半精度结论。
