# siliconflow-img-proxy

将 硅基流动生图 API 转成 OpenAI 兼容接口，可用于 Cherry Studio、OneAPI/NewAPI 等聚合平台，直接在聊天里以 Markdown 形式显示图片。

/v1/images/generations：OpenAI 兼容生图

/v1/chat/completions：Markdown 图片输出（![image](URL) 或 data:URI）

独立访问鉴权：ALLOW_CLIENT_KEY(S)（与上游 SILICONFLOW_API_KEY 完全分离）

可配置 /v1/models：通过环境变量提供模型清单

多图：n 顺序多次；或 sf_batch_size（若上游模型支持批量）

自定义参数：sf_seed、sf_num_steps、sf_guidance_scale、sf_cfg、sf_negative_prompt、sf_image_size、sf_batch_size

# 🔐 环境变量


| 变量名                                      | 是否必填 | 说明                                                                                                                               |
| ---------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------- |
| `SILICONFLOW_API_KEY`                    | ✅    | 上游硅基流动 Bearer Key（仅服务端使用，客户端不能覆盖）                                                                                                |
| `SF_BASE`                                | 选填   | 上游地址，默认 `https://api.siliconflow.cn/v1`                                                                                          |
| `PROVIDER_LABEL`                         | 选填   | `/v1/chat` 顶层 `provider` 字段，默认 `SiliconFlow`                                                                                     |
| `ALLOW_CLIENT_KEY` / `ALLOW_CLIENT_KEYS` | ✅*   | **代理访问密钥**（一个或多个，逗号/空白分隔）。任一存在即开启鉴权；未配置则公开（不建议）。                                                                                 |
| `MODELS_JSON`                            | 选填   | `/v1/models` 的 JSON 数组，如：`[{"id":"Qwen/Qwen-Image"},{"id":"Kwai-Kolors/Kolors"}]`，也可用 `["Qwen/Qwen-Image","Kwai-Kolors/Kolors"]` |
| `MODELS`                                 | 选填   | `/v1/models` 的逗号/空白分隔清单，如：`Qwen/Qwen-Image, Kwai-Kolors/Kolors`                                                                  |
# 🧪 API
## 1) 生成图片（OpenAI 兼容）

POST /v1/images/generations

```bash
curl -X POST "https://<worker>/v1/images/generations" \
  -H "Authorization: Bearer <proxy_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "a vintage camera on a wooden desk, warm light",
    "size": "1024x1024",
    "n": 2,
    "response_format": "url",
    "sf_seed": 1234567890,
    "sf_num_steps": 28,
    "sf_guidance_scale": 7.5,
    "sf_cfg": 3.5,
    "sf_negative_prompt": "low quality",
    "sf_batch_size": 2
  }'

```

多图策略

若设置 sf_batch_size 且模型支持批量，且 n=1：一次批量返回多张；

否则按 n 次数顺序多次调用上游返回多张。



## 2) 聊天出图（Markdown 渲染）

POST /v1/chat/completions

多轮对话：总是取 messages 中最新的 user 消息为 prompt

输出：message.content 为 Markdown 图片

response_format=url：`![image](https://...)`

response_format=b64_json：`![image](data:image/png;base64,...)`
```bash
curl -X POST "https://<worker>/v1/chat/completions" \
  -H "Authorization: Bearer <proxy_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "messages": [
      {"role":"system","content":"You are a helpful AI."},
      {"role":"user","content":"画两张蓝色机器人，木质桌面，冷色调"}
    ],
    "n": 2,
    "size": "1024x1024"
  }'

```


## GET /v1/models
通过`MODELS`等环境变量配置