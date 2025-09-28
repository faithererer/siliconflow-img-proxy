/**
 * SiliconFlow Images -> OpenAI-compatible Proxy (Cloudflare Worker • Classic)
 *
 * Endpoints:
 *   - POST /v1/images/generations
 *   - POST /v1/chat/completions   // message.content 为 Markdown 图片：![image](URL 或 data:URI)
 *   - GET  /v1/models             // 可配置模型清单（见环境变量）
 *   - OPTIONS /*                  // 动态 CORS 预检
 *
 * 关键点：
 *   - 访问鉴权：使用 ALLOW_CLIENT_KEY / ALLOW_CLIENT_KEYS（一个或多个，逗号/空白分隔）。
 *     -> 客户端必须以 Authorization: Bearer <proxy_key>（或 ?access_token=）访问本代理。
 *   - 上游鉴权：仅使用服务端环境变量 SILICONFLOW_API_KEY，**不接受客户端覆盖**。
 *   - /v1/models 可配置：MODELS_JSON（JSON 数组）或 MODELS（逗号分隔模型 id）。
 *   - /v1/chat：取 messages 中“最新的 user”消息为 prompt；返回 Markdown 图片（Cherry 可直接渲染）。
 *   - 多图：`n`（顺序多次）或 `sf_batch_size`（若上游模型支持批量）。
 *   - 自定义参数：sf_seed / sf_num_steps / sf_guidance_scale / sf_cfg /
 *                sf_negative_prompt / sf_image_size / sf_batch_size
 */

const DEFAULT_SF_BASE = "https://api.siliconflow.cn/v1";
const DEFAULT_PROVIDER_LABEL = "SiliconFlow";

/* --------------------------- Entry --------------------------- */

addEventListener("fetch", (event) => {
  event.respondWith(handleRequest(event));
});

async function handleRequest(event) {
  const req = event.request;
  const url = new URL(req.url);
  const path = url.pathname.replace(/\/+$/, "");
  const env = getEnv();

  // --- CORS 预检：动态回显客户端请求头 ---
  if (req.method === "OPTIONS") {
    const acrh = req.headers.get("Access-Control-Request-Headers") || "*";
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": acrh,
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin, Access-Control-Request-Headers",
      },
    });
  }

  // --- 代理访问鉴权（与上游 key 完全独立） ---
  const auth = authorizeProxy(req, env);
  if (!auth.ok) return auth.resp;

  try {
    if (path === "/v1/images/generations" && req.method === "POST") {
      const body = await safeJson(req);
      return imagesGenerations(body, req, env);
    }

    if (path === "/v1/chat/completions" && req.method === "POST") {
      const body = await safeJson(req);
      return chatCompletions(body, req, env);
    }

    if (path === "/v1/models" && req.method === "GET") {
      return listModels(env);
    }

    return json(
      { error: { message: `No route for ${req.method} ${path}`, type: "invalid_request_error" } },
      404
    );
  } catch (err) {
    return json(
      { error: { message: (err && err.message) || String(err), type: "internal_error" } },
      500
    );
  }
}

/* ------------------ /v1/images/generations ------------------- */

async function imagesGenerations(openaiBody, req, env) {
  const response_format = openaiBody?.response_format || "url";
  const upstreamKey = getUpstreamKey(env); // 仅用服务端上游 key
  const sfBase = env.SF_BASE || DEFAULT_SF_BASE;

  // model 必填（原样透传）
  const sfModel = openaiBody?.model ? String(openaiBody.model) : null;
  if (!sfModel) return json({ error: { message: "model is required", type: "invalid_request_error" } }, 400);

  // prompt 必填
  if (!openaiBody?.prompt || typeof openaiBody.prompt !== "string") {
    return json({ error: { message: "prompt is required (string)", type: "invalid_request_error" } }, 400);
  }

  const commonSfPayload = { model: sfModel, prompt: openaiBody.prompt };

  // size -> image_size（不支持的模型由上游处理）
  const imageSize = openaiBody?.sf_image_size || openaiBody?.size;
  if (typeof imageSize === "string") commonSfPayload.image_size = imageSize;

  // 自定义参数透传
  copyIfPresent(openaiBody, commonSfPayload, [
    ["sf_negative_prompt", "negative_prompt"],
    ["negative_prompt", "negative_prompt"],
    ["sf_num_steps", "num_inference_steps"],
    ["num_inference_steps", "num_inference_steps"],
    ["sf_guidance_scale", "guidance_scale"],
    ["guidance_scale", "guidance_scale"],
    ["sf_cfg", "cfg"],
    ["cfg", "cfg"],
    ["sf_seed", "seed"],
    ["seed", "seed"],
    ["sf_batch_size", "batch_size"],
  ]);

  const n = clampInt(openaiBody?.n ?? 1, 1, 10);
  const out = [];

  if (commonSfPayload.batch_size && n === 1) {
    // 支持批量的模型（如 Kolors）
    const r = await callSiliconFlow(commonSfPayload, upstreamKey, sfBase);
    const urls = (r?.images || []).map((it) => it?.url).filter(Boolean);
    if (response_format === "b64_json") {
      for (const u of urls) out.push({ b64_json: await fetchAndB64(u) });
    } else {
      for (const u of urls) out.push({ url: u });
    }
  } else {
    // 顺序多次调用实现 n 张
    for (let i = 0; i < n; i++) {
      const payload = { ...commonSfPayload };
      if (!("seed" in payload)) payload.seed = randSeed();
      const r = await callSiliconFlow(payload, upstreamKey, sfBase);
      const url = r?.images?.[0]?.url;
      if (!url) throw new Error("Upstream did not return image url");
      if (response_format === "b64_json") out.push({ b64_json: await fetchAndB64(url) });
      else out.push({ url });
    }
  }

  return json({ created: Math.floor(Date.now() / 1000), data: out });
}

/* ------------------- /v1/chat/completions -------------------- */

async function chatCompletions(openaiBody, req, env) {
  const messages = Array.isArray(openaiBody?.messages) ? openaiBody.messages : [];
  if (!messages.length) {
    return json({ error: { message: "messages is required (array)", type: "invalid_request_error" } }, 400);
  }

  // 取“最新的 user”作为 prompt
  const lastUser = [...messages].reverse().find((m) => m?.role === "user");
  const prompt =
    typeof lastUser?.content === "string" ? lastUser.content : flattenContentToText(lastUser?.content);
  if (!prompt) {
    return json({ error: { message: "No user prompt found in messages", type: "invalid_request_error" } }, 400);
  }

  // 原样透传 model
  const sfModel = openaiBody?.model ? String(openaiBody.model) : null;
  if (!sfModel) return json({ error: { message: "model is required", type: "invalid_request_error" } }, 400);

  const upstreamKey = getUpstreamKey(env); // 仅用服务端上游 key
  const sfBase = env.SF_BASE || DEFAULT_SF_BASE;
  const response_format = openaiBody?.response_format || "url";

  const commonSfPayload = { model: sfModel, prompt };
  const imageSize = openaiBody?.sf_image_size || openaiBody?.size;
  if (typeof imageSize === "string") commonSfPayload.image_size = imageSize;

  copyIfPresent(openaiBody, commonSfPayload, [
    ["sf_negative_prompt", "negative_prompt"],
    ["negative_prompt", "negative_prompt"],
    ["sf_num_steps", "num_inference_steps"],
    ["num_inference_steps", "num_inference_steps"],
    ["sf_guidance_scale", "guidance_scale"],
    ["guidance_scale", "guidance_scale"],
    ["sf_cfg", "cfg"],
    ["cfg", "cfg"],
    ["sf_seed", "seed"],
    ["seed", "seed"],
    ["sf_batch_size", "batch_size"],
  ]);

  const n = clampInt(openaiBody?.n ?? 1, 1, 10);
  const urls = [];

  if (commonSfPayload.batch_size && n === 1) {
    const r = await callSiliconFlow(commonSfPayload, upstreamKey, sfBase);
    for (const it of r?.images || []) if (it?.url) urls.push(it.url);
  } else {
    for (let i = 0; i < n; i++) {
      const payload = { ...commonSfPayload };
      if (!("seed" in payload)) payload.seed = randSeed();
      const r = await callSiliconFlow(payload, upstreamKey, sfBase);
      const url = r?.images?.[0]?.url;
      if (url) urls.push(url);
    }
  }

  // Markdown 输出；若 b64_json 则返回 data:URI 的 Markdown
  const id = `gen-${Math.floor(Date.now() / 1000)}-${cryptoRandomId(20)}`;
  let md;
  if (response_format === "b64_json") {
    const b64s = [];
    for (const u of urls) b64s.push(await fetchAndB64(u));
    md = b64s.map((b) => `![image](data:image/png;base64,${b})`).join("\n\n");
  } else {
    md = urls.map((u) => `![image](${u})`).join("\n\n");
  }

  const out = {
    id,
    provider: env.PROVIDER_LABEL || DEFAULT_PROVIDER_LABEL,
    model: openaiBody.model,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    choices: [
      {
        logprobs: null,
        finish_reason: "stop",
        native_finish_reason: "stop",
        index: 0,
        message: {
          role: "assistant",
          content: md,
          refusal: null,
          reasoning: null,
        },
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      prompt_tokens_details: null,
    },
  };

  return json(out);
}

/* ----------------------- /v1/models 可配置 ------------------- */

function listModels(env) {
  // 优先 MODELS_JSON：JSON 数组（字符串或 {id: "..."} 对象）
  if (env.MODELS_JSON) {
    try {
      const arr = JSON.parse(String(env.MODELS_JSON));
      const data = (Array.isArray(arr) ? arr : []).map(normalizeModelItem);
      return json({ object: "list", data });
    } catch {
      return json(
        { error: { message: "Invalid MODELS_JSON (must be JSON array)", type: "invalid_request_error" } },
        400
      );
    }
  }

  // 次选 MODELS：逗号/空白分隔
  if (env.MODELS) {
    const ids = String(env.MODELS)
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const data = ids.map((id) => normalizeModelItem(id));
    return json({ object: "list", data });
  }

  // 默认空列表
  return json({ object: "list", data: [] });
}

function normalizeModelItem(it) {
  const id = typeof it === "string" ? it : (it && (it.id || it.name || it.model)) || "unknown";
  return {
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "system",
  };
}

/* --------------------------- Auth ---------------------------- */

/**
 * 代理访问鉴权（独立密钥）：
 * - 若未配置 ALLOW_CLIENT_KEY(S)，则不鉴权（公开访问）。
 * - 若配置了任一，则客户端必须携带“代理访问密钥”，否则 401。
 *   位置：
 *     - Authorization: Bearer <proxy_key>（推荐）
 *     - 或 URL 查询参数 access_token / key / token / apikey
 */
function authorizeProxy(req, env) {
  const allowed = new Set();
  addKeys(allowed, env.ALLOW_CLIENT_KEY);
  addKeys(allowed, env.ALLOW_CLIENT_KEYS);
  // 兼容老习惯：也接收 ACCESS_TOKEN(S)/ACCESS_KEYS（可去掉）
  addKeys(allowed, env.ACCESS_TOKEN);
  addKeys(allowed, env.ACCESS_TOKENS);
  addKeys(allowed, env.ACCESS_KEYS);

  if (allowed.size === 0) return { ok: true }; // 未配置 => 公开

  let provided = req.headers.get("Authorization") || "";
  if (provided) provided = provided.replace(/^Bearer\s+/i, "").trim();

  if (!provided) {
    const url = new URL(req.url);
    provided =
      url.searchParams.get("access_token") ||
      url.searchParams.get("key") ||
      url.searchParams.get("token") ||
      url.searchParams.get("apikey") ||
      "";
    provided = provided.trim();
  }

  if (provided && allowed.has(provided)) return { ok: true };

  return {
    ok: false,
    resp: json(
      { error: { message: "Unauthorized: missing/invalid access token", type: "authentication_error" } },
      401,
      { "WWW-Authenticate": 'Bearer realm="proxy", error="invalid_token"' }
    ),
  };
}

function addKeys(set, raw) {
  if (!raw) return;
  String(raw)
    .split(/[,\n\r\t ]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .forEach((x) => set.add(x));
}

/* --------------------------- Helpers ------------------------- */

function json(obj, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "*",
      "Vary": "Origin, Access-Control-Request-Headers",
      ...extraHeaders,
    },
  });
}

async function safeJson(req) {
  const t = await req.text();
  try { return t ? JSON.parse(t) : {}; } catch { throw new Error("Invalid JSON body"); }
}

function clampInt(v, min, max) {
  const n = parseInt(String(v), 10);
  if (Number.isNaN(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function copyIfPresent(src, dst, pairs) {
  for (const [from, to] of pairs) if (src && src[from] !== undefined) dst[to] = src[from];
}

function randSeed() { return Math.floor(Math.random() * 1e10); }

function flattenContentToText(content) {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((c) => (typeof c === "string" ? c : c?.text || "")).join("\n");
  if (typeof content === "object" && content.text) return String(content.text);
  return "";
}

function cryptoRandomId(len) {
  const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  let out = "";
  const arr = new Uint8Array(len);
  crypto.getRandomValues(arr);
  for (const n of arr) out += alphabet[n % alphabet.length];
  return out;
}

function getEnv() {
  const g = globalThis;
  return {
    // 上游（只使用服务端 key，不允许客户端覆盖）
    SILICONFLOW_API_KEY: g.SILICONFLOW_API_KEY,
    SF_BASE: g.SF_BASE,

    // 返回中的 provider 名称
    PROVIDER_LABEL: g.PROVIDER_LABEL,

    // 代理访问密钥（独立于上游 key）
    ALLOW_CLIENT_KEY: g.ALLOW_CLIENT_KEY,
    ALLOW_CLIENT_KEYS: g.ALLOW_CLIENT_KEYS,
    // 兼容老变量名（可不配）
    ACCESS_TOKEN: g.ACCESS_TOKEN,
    ACCESS_TOKENS: g.ACCESS_TOKENS,
    ACCESS_KEYS: g.ACCESS_KEYS,

    // /v1/models 配置
    MODELS_JSON: g.MODELS_JSON,
    MODELS: g.MODELS,
  };
}

function getUpstreamKey(env) {
  if (!env.SILICONFLOW_API_KEY) throw new Error("SILICONFLOW_API_KEY not configured");
  return env.SILICONFLOW_API_KEY;
}

async function callSiliconFlow(payload, upstreamKey, sfBase) {
  const url = `${(sfBase || "").replace(/\/$/, "") || DEFAULT_SF_BASE}/images/generations`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${upstreamKey}` },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`Upstream ${r.status} ${r.statusText}: ${text}`);
  }
  return await r.json();
}

async function fetchAndB64(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Fetch image failed: ${r.status}`);
  const buf = await r.arrayBuffer();
  return arrayBufferToBase64(buf);
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, Array.from(chunk));
  }
  return btoa(binary);
}
