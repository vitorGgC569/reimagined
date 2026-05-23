/**
 * OContabil API client — substitui mock-data.js conectando ao backend real.
 *
 * Uso:
 *   import { api } from "./api-client.js";
 *   await api.login("admin", "admin");
 *   const clients = await api.listClients();
 *
 * Configuração: defina window.OCONTABIL_API_BASE no index.html ou via env build.
 */
const BASE = window.OCONTABIL_API_BASE || "http://localhost:8080/api";
const TOKEN_KEY = "ocontabil.token";

function authHeader() {
  const t = localStorage.getItem(TOKEN_KEY);
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...authHeader(),
      ...(opts.headers || {}),
    },
  });
  if (res.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    window.location.hash = "#/login";
    throw new Error("Sessão expirada");
  }
  if (!res.ok) {
    let err = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      err = body.message || err;
    } catch {
      /* ignore */
    }
    throw new Error(err);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  // ── Auth ──
  async login(username, password) {
    const out = await request("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    localStorage.setItem(TOKEN_KEY, out.token);
    localStorage.setItem("ocontabil.user", JSON.stringify(out));
    return out;
  },
  logout() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem("ocontabil.user");
  },
  currentUser() {
    try {
      return JSON.parse(localStorage.getItem("ocontabil.user") || "null");
    } catch {
      return null;
    }
  },
  isAuthenticated() {
    return !!localStorage.getItem(TOKEN_KEY);
  },

  // ── Clients ──
  listClients: () => request("/clients"),
  getClient: (id) => request(`/clients/${id}`),
  createClient: (data) =>
    request("/clients", { method: "POST", body: JSON.stringify(data) }),
  updateClient: (id, data) =>
    request(`/clients/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteClient: (id) => request(`/clients/${id}`, { method: "DELETE" }),

  // ── Documents ──
  listDocuments: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/documents${q ? "?" + q : ""}`);
  },
  getDocument: (id) => request(`/documents/${id}`),
  async uploadDocument(file, clientId, documentType) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("clientId", clientId);
    if (documentType) fd.append("documentType", documentType);
    const res = await fetch(`${BASE}/documents/upload`, {
      method: "POST",
      headers: authHeader(),
      body: fd,
    });
    if (!res.ok) {
      let err = `HTTP ${res.status}`;
      try {
        const b = await res.json();
        err = b.message || err;
      } catch {
        /* ignore */
      }
      throw new Error(err);
    }
    return res.json();
  },
  updateDocumentStatus: (id, status, extractedJson) =>
    request(`/documents/${id}/status`, {
      method: "PUT",
      body: JSON.stringify({ status, extractedJson }),
    }),

  // ── Polling de status de processamento ──
  async pollDocumentUntilProcessed(id, { intervalMs = 2000, timeoutMs = 120000 } = {}) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const d = await this.getDocument(id);
      if (d.status === "Validated" || d.status === "Error" || d.status === "ReadyForReview")
        return d;
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error("Timeout aguardando processamento");
  },

  // ── Health ──
  health: () => request("/health"),
};
