/**
 * SiliconFlow Images -> OpenAI-compatible Proxy (Cloudflare Worker • Classic)
 * - Edit 模型增强：若最新 user 消息未附图，自动使用“上一条 assistant 回复中的图片”作为输入图（支持 Markdown/data:URI）。
 *
 * Endpoints:
 *   - POST /v1/images/generations
 *   - POST /v1/chat/completions   // Markdown image output
 *   - GET  /v1/models             // configurable via env
 *   - OPTIONS /*                  // dynamic CORS
 *
 * Auth:
 *   - Proxy access keys: ALLOW_CLIENT_KEY / ALLOW_CLIENT_KEYS  (独立于上游，必带)
 *   - Upstream siliconflow key: SILICONFLOW_API_KEY (仅服务端；客户端不能覆盖)
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

  // CORS preflight (dynamic echo)
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

  // Proxy access auth
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
  const upstreamKey = getUpstreamKey(env);
  const sfBase = env.SF_BASE || DEFAULT_SF_BASE;

  const model = openaiBody?.model ? String(openaiBody.model) : null;
  if (!model) return json({ error: { message: "model is required", type: "invalid_request_error" } }, 400);

  if (!openaiBody?.prompt || typeof openaiBody.prompt !== "string") {
    return json({ error: { message: "prompt is required (string)", type: "invalid_request_error" } }, 400);
  }

  const commonSfPayload = { model, prompt: openaiBody.prompt };

  // size -> image_size
  const imageSize = openaiBody?.sf_image_size || openaiBody?.size;
  if (typeof imageSize === "string") commonSfPayload.image_size = imageSize;

  // 顶层 image / image_url / images[] 作为输入图（编辑模型必须）
  const isEdit = isImageEditModel(model);
  const img = pickImageFromBody(openaiBody);
  if (isEdit) {
    if (!img) {
      return json(
        { error: { message: "image is required for image-edit models (accepts data:image/... or URL, or image_url)", type: "invalid_request_error" } },
        400
      );
    }
    commonSfPayload.image = img;
  } else {
    if (img) commonSfPayload.image = img;
  }

  // 透传额外参数
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
    const r = await callSiliconFlow(commonSfPayload, upstreamKey, sfBase);
    const urls = (r?.images || []).map((it) => it?.url).filter(Boolean);
    if (response_format === "b64_json") {
      for (const u of urls) out.push({ b64_json: await fetchAndB64(u) });
    } else {
      for (const u of urls) out.push({ url: u });
    }
  } else {
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

  // 最新 user 文本
  const lastUser = [...messages].reverse().find((m) => m?.role === "user");
  const prompt =
    typeof lastUser?.content === "string" ? lastUser.content : flattenContentToText(lastUser?.content);
  if (!prompt) {
    return json({ error: { message: "No user prompt found in messages", type: "invalid_request_error" } }, 400);
  }

  const model = openaiBody?.model ? String(openaiBody.model) : null;
  if (!model) return json({ error: { message: "model is required", type: "invalid_request_error" } }, 400);

  const upstreamKey = getUpstreamKey(env);
  const sfBase = env.SF_BASE || DEFAULT_SF_BASE;
  const response_format = openaiBody?.response_format || "url";

  const commonSfPayload = { model, prompt };

  // —— 解析输入图 —— //
  // 1) 优先：最新 user 消息里的图片（OpenAI 风格/Markdown/data:URI）
  let img = pickImageFromMessage(lastUser);
  // 2) 兜底：顶层 body 的 image/image_url/images
  if (!img) img = pickImageFromBody(openaiBody);
  // 3) **新增** 再兜底（仅 edit 模型）：若还没有，则用“上一条 assistant 回复里的图片”
  const isEdit = isImageEditModel(model);
  if (isEdit && !img) {
    img = pickImageFromLastAssistant(messages);
  }

  if (isEdit) {
    if (!img) {
      return json(
        { error: { message: "image is required for image-edit models in chat (we also try last assistant image if user has none).", type: "invalid_request_error" } },
        400
      );
    }
    commonSfPayload.image = img;
  } else {
    if (img) commonSfPayload.image = img;
  }

  // 其它参数
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

  // Markdown 输出；b64_json 则 data:URI Markdown
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
    model,
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
  // token 用不上，固定 0
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      prompt_tokens_details: null,
    },
  };

  return json(out);
}

/* ----------------------- /v1/models (config) ----------------- */

function listModels(env) {
  if (env.MODELS_JSON) {
    try {
      const arr = JSON.parse(String(env.MODELS_JSON));
      const data = (Array.isArray(arr) ? arr : []).map(normalizeModelItem);
      return json({ object: "list", data });
    } catch {
      return json({ error: { message: "Invalid MODELS_JSON", type: "invalid_request_error" } }, 400);
    }
  }
  if (env.MODELS) {
    const ids = String(env.MODELS)
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const data = ids.map((id) => normalizeModelItem(id));
    return json({ object: "list", data });
  }
  return json({ object: "list", data: [] });
}

function normalizeModelItem(it) {
  const id = typeof it === "string" ? it : (it && (it.id || it.name || it.model)) || "unknown";
  return { id, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "system" };
}

/* --------------------------- Auth ---------------------------- */

function authorizeProxy(req, env) {
  const keys = new Set();
  addKeys(keys, env.ALLOW_CLIENT_KEY);
  addKeys(keys, env.ALLOW_CLIENT_KEYS);
  // 兼容早期命名（可保留/可移除）
  addKeys(keys, env.ACCESS_TOKEN);
  addKeys(keys, env.ACCESS_TOKENS);
  addKeys(keys, env.ACCESS_KEYS);

  if (keys.size === 0) return { ok: true }; // public

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

  if (provided && keys.has(provided)) return { ok: true };

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

/* ------------------------- Image parse ----------------------- */

/** 是否编辑模型：按名称包含 "Qwen/Qwen-Image-Edit" 来判断（可按需扩展） */
function isImageEditModel(model) {
  return /Qwen\/Qwen-Image-Edit/i.test(model);
}

/** 顶层 JSON 的 image/image_url/images */
function pickImageFromBody(body) {
  if (!body) return null;
  const v = body.image ?? body.image_url;
  const got = coerceImageStringOrObj(v);
  if (got) return got;

  const arr = body.images ?? body.image_urls;
  if (Array.isArray(arr)) {
    for (const item of arr) {
      const s = coerceImageStringOrObj(item);
      if (s) return s;
    }
  }
  return null;
}

/** 从消息（OpenAI chat 格式）提图片：content 数组的 image_url / 输入为 Markdown / data:URI / 裸 URL */
function pickImageFromMessage(msg) {
  if (!msg) return null;

  // 新版数组 content
  if (Array.isArray(msg.content)) {
    for (const it of msg.content) {
      if (it && it.type === "image_url") {
        const u = it.image_url?.url || it.image_url;
        const s = coerceImageStringOrObj(u);
        if (s) return s;
      }
      if (it && it.type === "input_image") {
        const u = it.image_url?.url || it.url || it.image;
        const s = coerceImageStringOrObj(u);
        if (s) return s;
      }
      if (typeof it === "string") {
        const s = extractImageFromPlainText(it);
        if (s) return s;
      }
      if (it && typeof it === "object") {
        const s = coerceImageStringOrObj(it.url || it.image || it.src || it.data || it.href);
        if (s) return s;
      }
    }
  }

  // 纯字符串 content：从 Markdown/裸 URL/data:URI 抽取
  if (typeof msg.content === "string") {
    const s = extractImageFromPlainText(msg.content);
    if (s) return s;
  }

  return null;
}

/** 关键新增：从“上一条 assistant”消息里提图片（Markdown/data:URI/数组 content） */
function pickImageFromLastAssistant(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === "assistant") {
      const found = pickImageFromMessage(m);
      if (found) return found;
    }
  }
  return null;
}

/** 接受 string 或 {url:string}，转成可用的 data:URI 或 http(s) URL */
function coerceImageStringOrObj(v) {
  if (!v) return null;
  if (typeof v === "string") return normalizeImageString(v);
  if (typeof v === "object" && v.url) return normalizeImageString(v.url);
  return null;
}

/** 标准化：允许 data:image/...;base64,... 或 http(s)://... */
function normalizeImageString(s) {
  const t = String(s).trim();
  if (/^data:image\//i.test(t)) return t;
  if (/^https?:\/\//i.test(t)) return t;
  return null;
}

/** 从纯文本中抽 url：匹配 ![...](url) 或裸 http(s) 或 data:image/... */
function extractImageFromPlainText(s) {
  if (!s) return null;
  const data = s.match(/data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/);
  if (data) return data[0];
  const md = s.match(/!\[[^\]]*\]\((https?:\/\/[^\s)]+)\)/);
  if (md) return md[1];
  const bare = s.match(/https?:\/\/[^\s)]+/);
  if (bare) return bare[0];
  return null;
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

function clampInt(v, min, max) { const n = parseInt(String(v), 10); if (Number.isNaN(n)) return min; return Math.max(min, Math.min(max, n)); }
function copyIfPresent(src, dst, pairs) { for (const [from, to] of pairs) if (src && src[from] !== undefined) dst[to] = src[from]; }
function randSeed() { return Math.floor(Math.random() * 1e10); }

function flattenContentToText(content) {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((c)=> (typeof c==="string"? c : c?.text || "")).join("\n");
  if (typeof content==="object" && content.text) return String(content.text);
  return "";
}

function cryptoRandomId(len) { const alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; let out=""; const arr=new Uint8Array(len); crypto.getRandomValues(arr); for (const n of arr) out+=alphabet[n%alphabet.length]; return out; }

function getEnv() {
  const g = globalThis;
  return {
    // upstream
    SILICONFLOW_API_KEY: g.SILICONFLOW_API_KEY,
    SF_BASE: g.SF_BASE,
    PROVIDER_LABEL: g.PROVIDER_LABEL,

    // proxy access keys (independent)
    ALLOW_CLIENT_KEY: g.ALLOW_CLIENT_KEY,
    ALLOW_CLIENT_KEYS: g.ALLOW_CLIENT_KEYS,
    // optional compat aliases
    ACCESS_TOKEN: g.ACCESS_TOKEN,
    ACCESS_TOKENS: g.ACCESS_TOKENS,
    ACCESS_KEYS: g.ACCESS_KEYS,

    // models
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
    if (!r.ok) throw new Error(`Fetch image failed: $ {
        r.status
    }`);
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
