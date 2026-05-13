// Cloudflare Worker — Sentry → GitHub relay
//
// Deploy steps (one-time, ~5 min):
//   1. Sign up free at cloudflare.com → Workers & Pages → Create Worker
//   2. Paste this file into the editor and click Deploy
//   3. Go to Worker Settings → Variables → add these secrets (encrypted):
//        SENTRY_WEBHOOK_SECRET  — copy from Sentry: Settings → Developer Settings → Internal Integrations → your webhook secret
//        GITHUB_PAT             — GitHub token with repo scope (github.com/settings/tokens)
//        GITHUB_REPO            — e.g.  sdev-sn/LLM_chat
//   4. Copy the Worker URL (e.g. https://sentry-relay.yourname.workers.dev)
//   5. In Sentry: Settings → Integrations → Webhooks → Add webhook → paste Worker URL
//      Set trigger: "Issue Created" and "Issue Resolved"

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const body = await request.text();

    // ── Validate Sentry HMAC signature ────────────────────────────────────
    const sentrySignature = request.headers.get("Sentry-Hook-Signature") || "";
    const isValid = await verifyHmac(body, env.SENTRY_WEBHOOK_SECRET, sentrySignature);
    if (!isValid) {
      return new Response("Unauthorized", { status: 401 });
    }

    // ── Parse payload — extract only non-sensitive fields ─────────────────
    let payload;
    try {
      payload = JSON.parse(body);
    } catch {
      return new Response("Bad request", { status: 400 });
    }

    const issue = payload?.data?.issue || {};
    const errorTitle = sanitize(issue.title || "Unknown error");
    const errorCulprit = sanitize(issue.culprit || "");
    const sentryUrl = sanitize(issue.web_url || "");
    const action = payload?.action || "created";

    // Only act on new issues, not resolved ones
    if (action !== "created") {
      return new Response("Ignored", { status: 200 });
    }

    // ── Dispatch to GitHub Actions ─────────────────────────────────────────
    const ghResponse = await fetch(
      `https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `token ${env.GITHUB_PAT}`,
          Accept: "application/vnd.github+json",
          "Content-Type": "application/json",
          "User-Agent": "sentry-relay-worker/1.0",
        },
        body: JSON.stringify({
          event_type: "sentry-error",
          client_payload: {
            error_title: errorTitle,
            error_culprit: errorCulprit,
            sentry_url: sentryUrl,
          },
        }),
      }
    );

    if (!ghResponse.ok) {
      const err = await ghResponse.text();
      console.error("GitHub dispatch failed:", err);
      return new Response("GitHub dispatch failed", { status: 502 });
    }

    return new Response("OK", { status: 200 });
  },
};

async function verifyHmac(body, secret, signature) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body));
  const hex = Array.from(new Uint8Array(mac))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return `sha256=${hex}` === signature;
}

// Strip any characters that could cause injection in downstream systems
function sanitize(str) {
  return String(str).replace(/[^a-zA-Z0-9 .,:\-_/()[\]#@]/g, "").slice(0, 200);
}
