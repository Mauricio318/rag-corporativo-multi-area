# -*- coding: utf-8 -*-
# RAG Corporativo — v4.0
# Melhorias aplicadas:
# - chromadb.Client (depreciado) → chromadb.PersistentClient
# - _session_id() computado uma vez por sessão (não muda entre eventos)
# - max_tokens: 400 → 1500 (respostas completas)
# - Proteção contra força bruta no login (máx. 5 tentativas)
# - Streaming de respostas com st.write_stream
# - PyPDF2 (abandonado) → pypdf
# - Cache de busca com TTL simples (invalida ao indexar novo doc)
# - construir_prototipos_por_area: assinatura compatível com cache
# - Sanitização básica de entrada (prompt injection mitigation)
# - Edição de settings pela própria UI do Admin

import os
import io
import re
import json
import time
import zipfile
import hashlib
import unicodedata
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import streamlit as st
import bcrypt
import extra_streamlit_components as stx

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from pypdf import PdfReader                          # fix: PyPDF2 → pypdf

from docx import Document as DocxDocument

from sentence_transformers import SentenceTransformer

import chromadb

st.set_page_config(page_title="RAG Corporativo", page_icon="🤖", layout="wide")

# =========================
# Constantes / caminhos
# =========================
PERSIST_DIR = "chroma_db"
DATA_DIR = Path("knowledge_files")
ARQUIVO_USUARIOS = Path("usuarios.json")
ARQUIVO_AREAS = Path("areas.json")
ARQUIVO_SETTINGS = Path("settings.json")
ARQUIVO_KNOW_INDEX = Path("knowledge_index.json")
ARQUIVO_AUDIT = Path("audit_log.jsonl")

COMPATIBILIDADE_MINIMA = 0.70
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300  # 5 minutos

# =========================
# Defaults
# =========================
DEFAULT_SETTINGS = {
    "__comments": {
        "TEMPERATURE": "Temperatura do LLM (0.0-1.0). Menor = respostas mais estáveis.",
        "TOP_K_CHAT": "Quantidade de trechos recuperados por pergunta no modo normal.",
        "LIMITE_CTX": "Tamanho máximo de caracteres do contexto (RAG) por resposta.",
        "CHUNK_SIZE": "Tamanho do chunk de texto ao indexar documentos.",
        "OVERLAP": "Sobreposição entre chunks (em caracteres).",
        "ENABLE_GLOBAL_SEARCH": "Se True, habilita a opção de busca 'Global' além de 'Área'.",
        "MAX_HISTORY_TURNS": "Número de pares usuário/assistente mantidos no histórico curto.",
        "FAST_MODE": "Se True, usa parâmetros mais rápidos no modo Usuário.",
        "SHOW_ADVANCED_API": "Se True, mostra campo de API Key na UI do Admin.",
        "MAX_TOKENS": "Máximo de tokens na resposta do LLM.",
        "LLM_MODEL": "Modelo do Google desejado (ex: gemini-2.5-flash)."
    },
    "LLM_MODEL": "gemini-2.5-flash",
    "TEMPERATURE": 0.2,
    "TOP_K_CHAT": 3,
    "LIMITE_CTX": 5000,
    "CHUNK_SIZE": 1000,
    "OVERLAP": 100,
    "ENABLE_GLOBAL_SEARCH": False,
    "MAX_HISTORY_TURNS": 4,
    "FAST_MODE": True,
    "SHOW_ADVANCED_API": False,
    "MAX_TOKENS": 1500,  # fix: era 400 — cortava respostas no meio
}

DEFAULT_AREAS = {
    "__comments": {"areas": "Lista de áreas. 'ativo' controla exibição; 'sementes' valida compatibilidade."},
    "areas": [
        {"nome": "RH", "ativo": True, "sementes": ["políticas de recursos humanos, benefícios, férias, folha de pagamento"]},
        {"nome": "Suporte", "ativo": True, "sementes": ["atendimento de TI, VPN, reset de senha, troubleshooting"]},
        {"nome": "Financeiro", "ativo": True, "sementes": ["reembolso, notas fiscais, orçamento, contabilidade"]},
    ]
}

DEFAULT_USUARIOS = {
    "__comments": {
        "users": "Lista de usuários. 'role' pode ser 'admin'. Senhas armazenadas como bcrypt.",
        "fields": "username, password_hash, role, ativo, criado_em (ISO)."
    },
    "users": []
}

DEFAULT_KNOW_INDEX = {
    "__comments": {"areas": "Mapeia cada área para lista de arquivos indexados."},
    "areas": {}
}

# =========================
# Helpers JSON
# =========================
def _safe_save_json(path: Path, data: Any):
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        bk = path.with_suffix(".bak.json")
        if path.exists():
            path.replace(bk)
        tmp.replace(path)
    except Exception as e:
        st.error(f"Falha ao salvar {path.name}: {e}")

def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        _safe_save_json(path, default)
        return json.loads(json.dumps(default))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            bk = path.with_suffix(".bak.json")
            if bk.exists():
                with open(bk, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return json.loads(json.dumps(default))

def _ensure_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for arq, default in [
        (ARQUIVO_SETTINGS, DEFAULT_SETTINGS),
        (ARQUIVO_AREAS, DEFAULT_AREAS),
        (ARQUIVO_USUARIOS, DEFAULT_USUARIOS),
        (ARQUIVO_KNOW_INDEX, DEFAULT_KNOW_INDEX),
    ]:
        if not arq.exists():
            _safe_save_json(arq, default)
        else:
            data = _safe_load_json(arq, default)
            for k, v in default.items():
                data.setdefault(k, v)
            _safe_save_json(arq, data)
    if not ARQUIVO_AUDIT.exists():
        ARQUIVO_AUDIT.write_text("", encoding="utf-8")

def _now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# fix: session_id computado UMA vez por sessão e armazenado
def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        base = f"{id(st)}-{datetime.utcnow().timestamp()}-{os.getpid()}"
        st.session_state.session_id = hashlib.sha1(base.encode()).hexdigest()[:12]
    return st.session_state.session_id

# =========================
# Auditoria
# =========================
def log_event(action: str, actor: Optional[str], details: Dict[str, Any]):
    record = {
        "ts": _now_iso(),
        "session": _get_session_id(),
        "actor": actor or "anon",
        "action": action,
        "details": details
    }
    try:
        with open(ARQUIVO_AUDIT, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"Falha ao gravar log: {e}")

# =========================
# Carregamento inicial
# =========================
_ensure_files()
S = _safe_load_json(ARQUIVO_SETTINGS, DEFAULT_SETTINGS)
# Garante MAX_TOKENS com valor mínimo seguro
S.setdefault("MAX_TOKENS", 1500)

# =========================
# Utils
# =========================
def slugificar(nome: str) -> str:
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def md5_texto(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def _sanitizar_entrada(texto: str) -> str:
    """Remove padrões comuns de prompt injection antes de enviar ao LLM."""
    proibidos = [
        r"ignore (all |previous |above )?instructions",
        r"(system|ignore|forget) prompt",
        r"jailbreak",
        r"act as (an? )?[a-z ]{3,30}without restrictions",
    ]
    t = texto
    for p in proibidos:
        t = re.sub(p, "[ENTRADA INVÁLIDA]", t, flags=re.IGNORECASE)
    return t[:4000]  # limite de tamanho

# =========================
# Autenticação + Proteção Brute-force
# =========================
def _carregar_usuarios():
    return _safe_load_json(ARQUIVO_USUARIOS, DEFAULT_USUARIOS)

def _salvar_usuarios(dados):
    _safe_save_json(ARQUIVO_USUARIOS, dados)

def _hash_senha(senha: str) -> str:
    return bcrypt.hashpw(senha.encode(), bcrypt.gensalt(rounds=12)).decode()

def _verificar_senha(senha: str, hash_salvo: str) -> bool:
    try:
        return bcrypt.checkpw(senha.encode(), hash_salvo.encode())
    except Exception:
        return False

def _login_bloqueado() -> bool:
    """Verifica se o IP/sessão está bloqueada por excesso de tentativas."""
    tentativas = st.session_state.get("login_attempts", 0)
    bloqueado_ate = st.session_state.get("login_locked_until", 0)
    if bloqueado_ate > time.time():
        restante = int(bloqueado_ate - time.time())
        st.error(f"🔒 Login bloqueado. Tente novamente em {restante}s.")
        return True
    return False

def _registrar_falha_login():
    tentativas = st.session_state.get("login_attempts", 0) + 1
    st.session_state.login_attempts = tentativas
    if tentativas >= MAX_LOGIN_ATTEMPTS:
        st.session_state.login_locked_until = time.time() + LOGIN_LOCKOUT_SECONDS
        st.error(f"🔒 Muitas tentativas. Login bloqueado por {LOGIN_LOCKOUT_SECONDS // 60} minutos.")
    else:
        st.error(f"Credenciais inválidas. ({tentativas}/{MAX_LOGIN_ATTEMPTS} tentativas)")

def _resetar_tentativas():
    st.session_state.login_attempts = 0
    st.session_state.login_locked_until = 0

def existe_admin() -> bool:
    data = _carregar_usuarios()
    return any(u.get("role") == "admin" and u.get("ativo", True) for u in data["users"])

def criar_admin(username: str, senha: str, actor: Optional[str] = None):
    if len(senha) < 8:
        raise ValueError("A senha deve ter no mínimo 8 caracteres.")
    data = _carregar_usuarios()
    if any(u["username"] == username for u in data["users"]):
        raise ValueError("Usuário já existe.")
    data["users"].append({
        "username": username,
        "password_hash": _hash_senha(senha),
        "role": "admin",
        "ativo": True,
        "criado_em": _now_iso()
    })
    _salvar_usuarios(data)
    log_event("CREATE_ADMIN", actor, {"username": username})

def atualizar_senha(username: str, senha_atual: str, nova_senha: str, actor: Optional[str] = None):
    if len(nova_senha) < 8:
        raise ValueError("Nova senha deve ter no mínimo 8 caracteres.")
    data = _carregar_usuarios()
    user = next((u for u in data["users"] if u["username"] == username and u.get("ativo", True)), None)
    if not user:
        raise ValueError("Usuário não encontrado/ativo.")
    if not _verificar_senha(senha_atual, user["password_hash"]):
        raise ValueError("Senha atual incorreta.")
    user["password_hash"] = _hash_senha(nova_senha)
    user["alterado_em"] = _now_iso()
    _salvar_usuarios(data)
    log_event("CHANGE_PASSWORD", actor or username, {"username": username})

# =========================
# Áreas CRUD
# =========================
def carregar_config_completa_areas():
    return _safe_load_json(ARQUIVO_AREAS, DEFAULT_AREAS)

def salvar_config_areas(cfg, actor=None, change_desc=None):
    _safe_save_json(ARQUIVO_AREAS, cfg)
    if change_desc:
        log_event("UPDATE_AREAS", actor, {"desc": change_desc})

def carregar_areas_ativas():
    cfg = carregar_config_completa_areas()
    return [a for a in cfg.get("areas", []) if a.get("ativo", True)]

def deletar_area(nome_area: str, actor=None):
    cfg = carregar_config_completa_areas()
    antes = len(cfg["areas"])
    cfg["areas"] = [a for a in cfg["areas"] if a["nome"] != nome_area]
    _safe_save_json(ARQUIVO_AREAS, cfg)

    area_dir = DATA_DIR / slugificar(nome_area)
    if area_dir.exists():
        for p in area_dir.glob("*"):
            try: p.unlink()
            except Exception: pass
        try: area_dir.rmdir()
        except Exception: pass

    try:
        client = get_chroma_client()
        cname = f"kb_{slugificar(nome_area)}"
        for col in client.list_collections():
            if col.name == cname:
                client.delete_collection(name=cname)
                break
    except Exception:
        pass

    idx = _safe_load_json(ARQUIVO_KNOW_INDEX, DEFAULT_KNOW_INDEX)
    if nome_area in idx.get("areas", {}):
        del idx["areas"][nome_area]
        _safe_save_json(ARQUIVO_KNOW_INDEX, idx)

    removidos = antes - len(cfg["areas"])
    if removidos > 0:
        log_event("DELETE_AREA", actor, {"area": nome_area})
    return removidos

# =========================
# Chroma + Embeddings + LLM
# fix: chromadb.Client(Settings(...)) depreciado → PersistentClient
# =========================
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=PERSIST_DIR)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def init_llm(model_name: str, temperature: float, max_tokens: int, api_key: str = None):
    if api_key and api_key.startswith("***"):
        api_key = None
    secret_key = None
    from pathlib import Path
    import os
    if Path(".streamlit/secrets.toml").exists():
        try:
            with open(".streamlit/secrets.toml", "r", encoding="utf-8") as fr:
                for linha in fr:
                    if linha.startswith("GOOGLE_API_KEY"):
                        secret_key = linha.split("=")[1].strip().strip('"')
                        break
        except Exception:
            pass
    key = api_key or os.getenv("GOOGLE_API_KEY") or secret_key
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=key
        )
        return llm
    except Exception as e:
        st.error(f"Erro ao inicializar LLM: {e}")
        return None

def colecao(area_nome: str, client=None):
    client = client or get_chroma_client()
    return client.get_or_create_collection(name=f"kb_{slugificar(area_nome)}")

# =========================
# Protótipos por área
# fix: converter lista para tuple para compatibilidade com cache
# =========================
@st.cache_resource(show_spinner=False)
def construir_prototipos_por_area(_areas_tuple, _embedder):
    """_areas_tuple: tuple de dicts (hashable para cache)."""
    protos = {}
    for a in _areas_tuple:
        sementes = a.get("sementes", [])
        if not sementes:
            continue
        embs = _embedder.encode(sementes, normalize_embeddings=True)
        protos[a["nome"]] = embs.mean(axis=0)
    return protos

# =========================
# Chunking + Leitura
# fix: PyPDF2 → pypdf
# =========================
@dataclass
class Chunk:
    texto: str
    arquivo: str
    pagina: int

def dividir_chunks(texto, arquivo, pagina, tam, overlap):
    t = " ".join(texto.split())
    if not t:
        return []
    out = []; i = 0
    while i < len(t):
        f = min(i + tam, len(t))
        out.append(Chunk(t[i:f], arquivo, pagina))
        i = f - overlap if (f - overlap) > i else f
    return out

def ler_pdf(b, nome, tam, overlap):
    out = []
    r = PdfReader(io.BytesIO(b))   # fix: pypdf.PdfReader
    for i, p in enumerate(r.pages):
        txt = p.extract_text() or ""
        out += dividir_chunks(txt, nome, i + 1, tam, overlap)
    return out

def ler_docx(b, nome, tam, overlap):
    d = DocxDocument(io.BytesIO(b))
    txt = "\n".join([p.text for p in d.paragraphs])
    return dividir_chunks(txt, nome, 1, tam, overlap)

def ler_txt(b, nome, tam, overlap):
    return dividir_chunks(b.decode("utf-8", "ignore"), nome, 1, tam, overlap)

# =========================
# Validação de compatibilidade (70%)
# =========================
def validar_compatibilidade(chunks, area, protos, embedder):
    if not chunks:
        return {"total": 0, "ratio": 1.0, "amostras_fora": []}
    textos = [c.texto for c in chunks]
    embs = embedder.encode(textos, normalize_embeddings=True)
    nomes = list(protos.keys())
    if not nomes or area not in protos:
        return {"total": len(chunks), "ratio": 1.0, "amostras_fora": []}
    matriz = np.vstack([protos[n] for n in nomes])
    sims = embs @ matriz.T
    best_idx = sims.argmax(axis=1)
    best_areas = [nomes[i] for i in best_idx]
    total = len(best_areas)
    match = sum(1 for a in best_areas if a == area)
    ratio = match / max(total, 1)
    amostras = []
    for i, a_ in enumerate(best_areas):
        if a_ != area and len(amostras) < 3:
            prev = textos[i][:200].replace("\n", " ")
            amostras.append({"detected_area": a_, "preview": prev + ("…" if len(textos[i]) > 200 else "")})
    return {"total": total, "ratio": ratio, "amostras_fora": amostras}

# =========================
# Indexação
# =========================
def _carregar_indice_conhecimento():
    return _safe_load_json(ARQUIVO_KNOW_INDEX, DEFAULT_KNOW_INDEX)

def _salvar_indice_conhecimento(idx):
    _safe_save_json(ARQUIVO_KNOW_INDEX, idx)

def _invalidar_cache_busca():
    """Invalida o cache de busca para forçar rebusca após novos uploads."""
    st.session_state.cache_busca = {}
    st.session_state.cache_busca_version = st.session_state.get("cache_busca_version", 0) + 1

def adicionar_ao_chroma(area, chunks, embedder, client=None):
    if not chunks:
        return 0, []
    client = client or get_chroma_client()
    col = colecao(area, client)
    textos = [c.texto for c in chunks]
    embs = embedder.encode(textos, normalize_embeddings=True)
    ids, metas = [], []
    for i, c in enumerate(chunks):
        h = md5_texto(c.texto)
        ids.append(f"{c.arquivo}|{c.pagina}|{i}|{h}")
        metas.append({"area": area, "arquivo": c.arquivo, "pagina": c.pagina, "file_hash": h})
    col.add(ids=ids, embeddings=embs.tolist(), documents=textos, metadatas=metas)
    return len(textos), list(set(m["file_hash"] for m in metas))

def _salvar_arquivo_area(area: str, nome: str, conteudo: bytes) -> Path:
    adir = DATA_DIR / slugificar(area)
    adir.mkdir(parents=True, exist_ok=True)
    destino = adir / nome
    with open(destino, "wb") as f:
        f.write(conteudo)
    return destino

def indexar_arquivo_bytes(area, nome, b, S, protos, embedder, actor=None, forcar_indexacao=False):
    ext = nome.lower().split(".")[-1]
    if ext == "pdf":
        chunks = ler_pdf(b, nome, S["CHUNK_SIZE"], S["OVERLAP"])
    elif ext == "docx":
        chunks = ler_docx(b, nome, S["CHUNK_SIZE"], S["OVERLAP"])
    elif ext == "txt":
        chunks = ler_txt(b, nome, S["CHUNK_SIZE"], S["OVERLAP"])
    else:
        return 0, 0.0, {"ok": False, "msg": f"Tipo não suportado: {nome}"}

    if not chunks:
        return 0, 1.0, {"ok": False, "msg": f"Nenhum texto extraído de: {nome}"}

    rv = validar_compatibilidade(chunks, area, protos, embedder)
    if rv["ratio"] < COMPATIBILIDADE_MINIMA and not forcar_indexacao:
        return 0, rv["ratio"], {
            "ok": False,
            "msg": f"Compatibilidade insuficiente ({rv['ratio']*100:.1f}%). Marque a caixa de Forçar para ignorar isto.",
            "amostras": rv["amostras_fora"]
        }

    # Remove versão anterior se existir para não duplicar
    deletar_documento_da_area(area, nome, "Substituição de arquivo", actor)

    qtd, file_hashes = adicionar_ao_chroma(area, chunks, embedder)
    caminho = _salvar_arquivo_area(area, nome, b)
    idx = _carregar_indice_conhecimento()
    idx.setdefault("areas", {}).setdefault(area, [])
    entry = {
        "filename": nome,
        "saved_path": str(caminho),
        "uploaded_at": _now_iso(),
        "size_bytes": len(b),
        "ext": ext,
        "doc_hash": md5_bytes(b)
    }
    idx["areas"][area].append(entry)
    _salvar_indice_conhecimento(idx)
    _invalidar_cache_busca()  # fix: invalida cache ao indexar
    log_event("UPLOAD_DOC", actor, {"area": area, "filename": nome, "size": len(b)})
    return qtd, rv["ratio"], {"ok": True, "msg": f"Indexado: {nome}", "hashes": file_hashes}

def deletar_documento_da_area(area, filename, reason, actor=None):
    try:
        client = get_chroma_client()
        col = colecao(area, client)
        try:
            col.delete(where={"arquivo": filename})
        except Exception:
            res = col.get(where={"arquivo": filename})
            if res and res.get("ids"):
                col.delete(ids=res["ids"])
    except Exception as e:
        st.warning(f"Não foi possível limpar Chroma para {filename}: {e}")

    idx = _carregar_indice_conhecimento()
    items = idx.get("areas", {}).get(area, [])
    novos = []
    removed = False
    removed_path = None
    for it in items:
        if it.get("filename") == filename:
            removed = True
            removed_path = it.get("saved_path", "")
            try:
                p = Path(removed_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        else:
            novos.append(it)
    if area in idx.get("areas", {}):
        idx["areas"][area] = novos
        _salvar_indice_conhecimento(idx)

    if removed:
        _invalidar_cache_busca()
        log_event("DELETE_DOC", actor, {"area": area, "filename": filename, "reason": reason, "path": removed_path})
    return removed

# =========================
# Busca com cache por sessão
# =========================
def _init_cache():
    if "cache_busca" not in st.session_state:
        st.session_state.cache_busca = {}

def _cache_get(chave):
    return st.session_state.cache_busca.get(chave)

def _cache_set(chave, valor):
    # Limita tamanho do cache (máx 200 entradas)
    if len(st.session_state.cache_busca) >= 200:
        st.session_state.cache_busca.clear()
    st.session_state.cache_busca[chave] = valor

def buscar_na_area(pergunta, area, top_k, embedder, client=None):
    client = client or get_chroma_client()
    col = colecao(area, client)
    q = embedder.encode([pergunta], normalize_embeddings=True).tolist()
    try:
        return col.query(query_embeddings=q, n_results=top_k)
    except Exception:
        return {}

def buscar_global(pergunta, areas_ativas, top_k, embedder, client=None):
    client = client or get_chroma_client()
    q = embedder.encode([pergunta], normalize_embeddings=True).tolist()
    agreg = []
    for a in areas_ativas:
        col = colecao(a, client)
        try:
            r = col.query(query_embeddings=q, n_results=top_k)
            if r and r.get("documents"):
                for i in range(len(r["documents"][0])):
                    agreg.append({
                        "area": a,
                        "doc": r["documents"][0][i],
                        "meta": r["metadatas"][0][i],
                        "id": r["ids"][0][i],
                        "dist": r["distances"][0][i] if "distances" in r else None
                    })
        except Exception:
            pass
    agreg.sort(key=lambda x: x["dist"] if x["dist"] is not None else 0.0)
    return agreg[:top_k]

# =========================
# RAG
# =========================
INSTRUCOES_SISTEMA = """Você é um assistente corporativo rigoroso. Responda em português do Brasil com clareza e objetividade.
- IMPORTANTE: Baseie-se EXCLUSIVAMENTE nas informações contidas no CONTEXTO fornecido e no HISTÓRICO de conversa para perguntas, dados, processos ou informações institucionais.
- Se a resposta não estiver no CONTEXTO para dúvidas sobre trabalho, NUNCA utilize conhecimentos externos para tentar ajudar. Diga APENAS: "Infelizmente eu não possuo essa informação nos documentos da base de conhecimento da nossa empresa."
- EXCEÇÃO DE IDENTIDADE: Se o usuário te der 'Bom dia' ou perguntar quem é você / como você ajuda, responda amigavelmente (mesmo sem contexto em anexo). Explique que você é a Inteligência Artificial corporativa projetada unicamente para ler, buscar e responder dúvidas sobre os documentos da Instituição baseada na Área atual.
- Nunca invente, adivinhe, ou traga dados e processos externos à sua empresa.
- Quando houver os dados solicitados no contexto, explique passo a passo e de forma organizada.
"""

def montar_contexto_area(res, limite_chars):
    if not res or not res.get("documents"):
        return "NENHUM CONTEXTO ENCONTRADO", "Sem fontes."
    docs = res["documents"][0]; metas = res["metadatas"][0]
    trechos = []; cites = []; total = 0
    for d, m in zip(docs, metas):
        header = f"[Fonte: {m.get('arquivo','?')} | pág. {m.get('pagina','?')}]"
        snippet = f"{header}\n{d}\n"
        if total + len(snippet) > limite_chars:
            break
        trechos.append(snippet)
        cites.append(f"- {m.get('arquivo','?')} (pág. {m.get('pagina','?')})")
        total += len(snippet)
    return ("\n\n".join(trechos) if trechos else "NENHUM CONTEXTO ENCONTRADO",
            "\n".join(cites) if cites else "Sem fontes.")

def montar_contexto_global(res_list, limite_chars):
    if not res_list:
        return "NENHUM CONTEXTO ENCONTRADO", "Sem fontes."
    trechos = []; cites = []; total = 0
    for r in res_list:
        m = r["meta"]
        header = f"[Fonte: {m.get('arquivo','?')} | pág. {m.get('pagina','?')} | área: {m.get('area','?')}]"
        snippet = f"{header}\n{r['doc']}\n"
        if total + len(snippet) > limite_chars:
            break
        trechos.append(snippet)
        cites.append(f"- {m.get('arquivo','?')} (pág. {m.get('pagina','?')}) — {m.get('area','?')}")
        total += len(snippet)
    return ("\n\n".join(trechos) if trechos else "NENHUM CONTEXTO ENCONTRADO",
            "\n".join(cites) if cites else "Sem fontes.")

def _recortar_historico(msgs, max_turns):
    if max_turns <= 0:
        return []
    pares = []; tmp = []
    for m in reversed(msgs):
        tmp.append(m)
        if len(tmp) == 2:
            pares.append(tmp); tmp = []
        if len(pares) >= max_turns:
            break
    rec = []
    for par in reversed(pares):
        rec.extend(reversed(par))
    return rec

def _montar_msgs_llm(pergunta, contexto, historico, S):
    sys_prompt = f"""{INSTRUCOES_SISTEMA}
DIRETRIZES TÉCNICAS RESTRITAS:
- Responda de forma natural e vá direto ao ponto. NUNCA inicie sua mensagem dizendo "Olá", "Oi" ou "Tudo bem" para perguntas normais que pertencem ao meio do chat.
- Mantenha todo o seu raciocínio contextualizado ao histórico logo abaixo e não se repita de forma robótica."""
    msgs = [SystemMessage(content=sys_prompt)]
    
    historico_curto = _recortar_historico(historico, S["MAX_HISTORY_TURNS"])
    for m in historico_curto:
        if m["papel"] == "usuario":
            msgs.append(HumanMessage(content=m["conteudo"]))
        else:
            msgs.append(AIMessage(content=m["conteudo"]))
            
    user_prompt = f"""# INFORMAÇÕES RESGATADAS (RAG)
{contexto}

# SOLICITAÇÃO
{pergunta}

DIRETRIZES PARA A RESPOSTA FINAL:
- Lembre-se vigorosamente de tudo o que concordamos e falamos no histórico fornecido.
- Traga um passo a passo claro, incluindo exemplos e alternativas quando houver.
- Aponte pré-requisitos e riscos comuns.
- Se faltarem dados no contexto, explique o que falta e como obtê-los."""
    msgs.append(HumanMessage(content=user_prompt))
    return msgs

def responder_com_rag(pergunta, escopo, area_atual, top_k, limite_ctx, llm, embedder, areas_ativas):
    """Retorna (generator_ou_str, fontes_str)."""
    _init_cache()
    pergunta_sanitizada = _sanitizar_entrada(pergunta)

    if escopo == "Área":
        chave = (f"AREA::{area_atual}", pergunta_sanitizada)
        res = _cache_get(chave)
        if res is None:
            res = buscar_na_area(pergunta_sanitizada, area_atual, top_k, embedder)
            _cache_set(chave, res)
        contexto, fontes = montar_contexto_area(res, limite_ctx)
    else:
        chave = ("GLOBAL", pergunta_sanitizada)
        res_list = _cache_get(chave)
        if res_list is None:
            res_list = buscar_global(pergunta_sanitizada, areas_ativas, top_k, embedder)
            _cache_set(chave, res_list)
        contexto, fontes = montar_contexto_global(res_list, limite_ctx)

    msgs = _montar_msgs_llm(pergunta_sanitizada, contexto, st.session_state.mensagens, S)
    rodape = f"\n\n---\n**Fontes consultadas (RAG):**\n{fontes if fontes else 'Sem fontes.'}"

    # fix: streaming com llm.stream()
    def _stream():
        for chunk in llm.stream(msgs):
            yield chunk.content
        yield rodape

    return _stream, fontes

# =========================
# UI helpers
# =========================
def nav_sidebar_admin():
    with st.sidebar:
        st.markdown("---")
        if "view" not in st.session_state:
            st.session_state.view = "Chat"
        options = ["Chat", "Uploads", "Conhecimento", "Admin", "Settings", "Logs"]
        st.session_state.view = st.radio("Navegar", options, index=options.index(st.session_state.view))

# =========================
# APP
# =========================
st.title("RAG Corporativo — Chat com seus documentos 📚🤖")

if "perfil" not in st.session_state:
    st.session_state.perfil = "Usuário"
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "show_admin_login" not in st.session_state:
    st.session_state.show_admin_login = False

try:
    cookie_manager = stx.CookieManager()
    if not st.session_state.get("usuario"):
        cookie_user = cookie_manager.get("admin_session")
        if cookie_user:
            try:
                import json
                if isinstance(cookie_user, str):
                    st.session_state.usuario = json.loads(cookie_user)
                else:
                    st.session_state.usuario = cookie_user
                st.session_state.perfil = "Admin" if st.session_state.usuario.get("role") == "admin" else "Usuário"
            except Exception:
                pass
except Exception:
    cookie_manager = None

cfg_areas = carregar_config_completa_areas()
areas_ativas = [a for a in cfg_areas["areas"] if a.get("ativo", True)]
nomes_areas_ativas = [a["nome"] for a in areas_ativas]
embedder = get_embedder()
# fix: converter para tuple para que @cache_resource funcione
prototipos = construir_prototipos_por_area(tuple(areas_ativas), embedder)

# ===== Barra lateral =====
with st.sidebar:
    st.markdown("### Configurações")
    
    def on_area_change():
        st.session_state.mensagens = []
        
    area_escolhida = st.selectbox("Área", options=nomes_areas_ativas,
                                  index=0 if nomes_areas_ativas else None, key="area_sidebar", on_change=on_area_change)
    escopo = "Área"
    if S["ENABLE_GLOBAL_SEARCH"]:
        escopo = st.radio("Escopo da busca", ["Área", "Global"], index=0, key="escopo_sidebar", on_change=on_area_change)

    st.markdown("---")
    if not existe_admin():
        st.warning("🚀 Primeira execução: crie o Admin Master.")
    if not st.session_state.get("usuario"):
        if st.button("Entrar como Admin", use_container_width=True):
            st.session_state.show_admin_login = True
    else:
        st.caption(f"Logado como **{st.session_state.usuario['username']}** ({st.session_state.usuario['role']})")
        if st.button("Sair", use_container_width=True):
            st.session_state.usuario = None
            st.session_state.perfil = "Usuário"
            if "cookie_manager" in globals() and cookie_manager:
                cookie_manager.delete("admin_session")

    if st.session_state.show_admin_login and not st.session_state.get("usuario"):
        st.subheader("Login Admin")
        lu = st.text_input("Usuário", key="login_user")
        lp = st.text_input("Senha", type="password", key="login_pass")
        colL, colR = st.columns(2)
        with colL:
            if st.button("Entrar"):
                if not _login_bloqueado():
                    data = _carregar_usuarios()
                    user = next((u for u in data["users"]
                                 if u["username"] == lu and u.get("ativo", True)), None)
                    if user and _verificar_senha(lp, user["password_hash"]):
                        st.session_state.usuario = {"username": user["username"], "role": user["role"]}
                        st.session_state.perfil = "Admin"
                        st.session_state.show_admin_login = False
                        if "cookie_manager" in globals() and cookie_manager:
                            import datetime, json
                            cookie_manager.set("admin_session", json.dumps(st.session_state.usuario), expires_at=datetime.datetime.now() + datetime.timedelta(days=7))
                        _resetar_tentativas()
                        st.success(f"Bem-vindo, {user['username']}!")
                    else:
                        _registrar_falha_login()
        with colR:
            if st.button("Cancelar"):
                st.session_state.show_admin_login = False

# Wizard master admin (primeiro uso)
if not existe_admin():
    st.info("Crie o usuário **Master** para administrar o sistema.")
    u = st.text_input("Usuário master")
    p1 = st.text_input("Senha (mín. 8 caracteres)", type="password")
    p2 = st.text_input("Confirme a senha", type="password")
    if st.button("Criar usuário master"):
        if not u or not p1:
            st.error("Usuário e senha são obrigatórios.")
        elif p1 != p2:
            st.error("As senhas não coincidem.")
        else:
            try:
                criar_admin(u.strip(), p1, actor="setup")
                st.success("Usuário master criado! Clique em 'Entrar como Admin' na lateral.")
            except Exception as e:
                st.error(str(e))
    st.stop()

# ===== Modo Usuário =====
if st.session_state.perfil == "Usuário" or not st.session_state.get("usuario"):
    top_k = 2 if S.get("FAST_MODE") else S["TOP_K_CHAT"]
    lim_ctx = 2800 if S.get("FAST_MODE") else S["LIMITE_CTX"]
    llm = init_llm(S.get("LLM_MODEL", "gemini-2.5-flash"), S["TEMPERATURE"], S["MAX_TOKENS"])
    if llm is None:
        st.warning("⚠️ O sistema está sem a chave da API do Google. Configure-a no menu lateral do painel Admin (Opções avançadas).")

    st.header("💬 Chat")
    for m in st.session_state.mensagens:
        with st.chat_message("user" if m["papel"] == "usuario" else "assistant"):
            st.write(m["conteudo"])

    entrada = st.chat_input("Digite sua pergunta…")
    if entrada:
        st.session_state.mensagens.append({"papel": "usuario", "conteudo": entrada})
        with st.chat_message("user"):
            st.write(entrada)
        with st.chat_message("assistant"):
            if llm is None:
                st.error("LLM indisponível.")
                resposta = "Não foi possível responder (LLM não inicializado)."
                st.write(resposta)
            else:
                try:
                    nomes_areas = [a["nome"] for a in areas_ativas]
                    stream_fn, fontes = responder_com_rag(
                        entrada, escopo, area_escolhida, top_k, lim_ctx, llm, embedder, nomes_areas
                    )
                    # fix: streaming real
                    resposta = st.write_stream(stream_fn())
                except Exception as e:
                    st.error(f"Erro no RAG: {e}")
                    resposta = "Ocorreu um erro ao responder."
                    st.write(resposta)
            st.session_state.mensagens.append({"papel": "assistente", "conteudo": resposta or ""})

# ===== Modo Admin =====
else:
    if st.session_state.usuario.get("role") != "admin":
        st.error("Somente administradores podem acessar este modo.")
        st.stop()

    with st.sidebar:
        st.markdown("### Opções avançadas")
        api_key_manual = None
        if S.get("SHOW_ADVANCED_API", False):
            import os
            from pathlib import Path
            
            saved_key = None
            if Path(".streamlit/secrets.toml").exists():
                try:
                    with open(".streamlit/secrets.toml", "r", encoding="utf-8") as fr:
                        for linha in fr:
                            if linha.startswith("GOOGLE_API_KEY"):
                                saved_key = linha.split("=")[1].strip().strip('"')
                except Exception:
                    pass
            
            masked_atual = ""
            if saved_key:
                masked_atual = f"***{saved_key[-4:]}" if len(saved_key) >= 4 else "***"
                
                if "api_test_ok" not in st.session_state:
                    try:
                        mdl = S.get("LLM_MODEL", "gemini-2.5-flash")
                        test_llm = ChatGoogleGenerativeAI(model=mdl, google_api_key=saved_key, max_tokens=10)
                        test_llm.invoke("Teste de conexão.")
                        st.session_state.api_test_ok = True
                        st.session_state.api_test_err = ""
                    except Exception as try_e:
                        st.session_state.api_test_ok = False
                        st.session_state.api_test_err = str(try_e)
                
                if st.session_state.api_test_ok:
                    st.success("✅ A chave configurada está funcionando.")
                else:
                    st.error("❌ A chave configurada apresenta um erro.")
                    if st.session_state.get("api_test_err"):
                        st.caption(f"Detalhe técnico: {st.session_state.api_test_err}")
            else:
                st.info("Nenhuma chave armazenada ainda.")

            api_key_manual = st.text_input(
                "Sua Google API Key" if saved_key else "Nova Google API Key", 
                value=masked_atual,
                type="password" if saved_key else "default",
                help="Apague tudo na caixa caso queira inserir uma nova." if saved_key else ""
            )
            
            if st.button("💾 Salvar/Atualizar API Key"):
                if api_key_manual and api_key_manual != masked_atual:
                    old_key_masked = masked_atual if saved_key else "Nenhuma"
                    os.makedirs(".streamlit", exist_ok=True)
                    with open(".streamlit/secrets.toml", "w", encoding="utf-8") as f:
                        f.write(f'GOOGLE_API_KEY = "{api_key_manual}"\n')
                        
                    new_key_masked = f"***{api_key_manual[-4:]}" if len(api_key_manual) >= 4 else "***"
                    usr_nome = st.session_state.usuario.get("username", "admin")
                    log_event("UPDATE_API_KEY", usr_nome, {"antiga": old_key_masked, "nova": new_key_masked})
                    
                    st.session_state.pop("api_test_ok", None)
                    st.session_state.pop("api_test_err", None)
                    init_llm.clear()
                    st.success("Configuração atualizada! Pressione 'Rerun'.")
                elif api_key_manual == masked_atual and masked_atual != "":
                    st.info("A chave na caixa é a mesma salva atualmente.")
                else:
                    st.warning("Insira uma chave antes de salvar.")

    nav_sidebar_admin()

    # --- Chat Admin ---
    if st.session_state.view == "Chat":
        top_k = S["TOP_K_CHAT"]
        lim_ctx = S["LIMITE_CTX"]
        # Testar com chave da UI, se houver
        real_api_key_manual = api_key_manual if (api_key_manual and not api_key_manual.startswith("***")) else None
        llm = init_llm(S.get("LLM_MODEL", "gemini-2.5-flash"), S["TEMPERATURE"], S["MAX_TOKENS"], real_api_key_manual)
        if llm is None:
            st.warning("⚠️ Configure a Google API Key no menu lateral esquerdo.")
        st.header("💬 Chat (Admin)")
        for m in st.session_state.mensagens:
            with st.chat_message("user" if m["papel"] == "usuario" else "assistant"):
                st.write(m["conteudo"])
        entrada = st.chat_input("Digite sua pergunta…")
        if entrada:
            st.session_state.mensagens.append({"papel": "usuario", "conteudo": entrada})
            with st.chat_message("user"):
                st.write(entrada)
            with st.chat_message("assistant"):
                if llm is None:
                    st.error("LLM indisponível.")
                    resposta = "Não foi possível responder."
                    st.write(resposta)
                else:
                    try:
                        nomes_areas = [a["nome"] for a in areas_ativas]
                        stream_fn, fontes = responder_com_rag(
                            entrada, escopo, area_escolhida, top_k, lim_ctx, llm, embedder, nomes_areas
                        )
                        resposta = st.write_stream(stream_fn())
                    except Exception as e:
                        st.error(f"Erro no RAG: {e}")
                        resposta = "Ocorreu um erro ao responder."
                        st.write(resposta)
                st.session_state.mensagens.append({"papel": "assistente", "conteudo": resposta or ""})

    # --- Uploads ---
    elif st.session_state.view == "Uploads":
        st.header("📤 Upload de documentos (Admin)")
        if "file_uploader_key" not in st.session_state:
            st.session_state.file_uploader_key = 1
            
        area_up = st.selectbox("Área de destino", options=nomes_areas_ativas, key="up_area")
        forcar_up = st.checkbox("Forçar indexação (ignorar análise de compatibilidade)", value=False)
        arquivos = st.file_uploader("Selecione arquivos (PDF/DOCX/TXT/ZIP)",
                                    type=["pdf", "docx", "txt", "zip"],
                                    accept_multiple_files=True, key=f"up_files_{st.session_state.file_uploader_key}")
        if st.button("Processar Arquivos Selecionados") and arquivos and area_up:
            total = 0
            progress = st.progress(0, text="Processando…")
            for idx_f, f in enumerate(arquivos):
                progress.progress((idx_f) / len(arquivos), text=f"Processando {f.name}…")
                nome = f.name; b = f.read()
                if nome.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(b)) as zf:
                            for info in zf.infolist():
                                if info.is_dir():
                                    continue
                                ext = info.filename.lower().split(".")[-1]
                                if ext not in ["pdf", "docx", "txt"]:
                                    continue
                                ib = zf.read(info.filename)
                                inome = os.path.basename(info.filename)
                                qtd, ratio, infores = indexar_arquivo_bytes(
                                    area_up, inome, ib, S, prototipos, embedder,
                                    actor=st.session_state.usuario["username"], forcar_indexacao=forcar_up
                                )
                                if not infores["ok"]:
                                    st.warning(f"{inome}: {infores['msg']}")
                                else:
                                    total += qtd
                                    st.success(f"{infores['msg']} | compat.: {ratio*100:.1f}% | chunks: {qtd}")
                    except zipfile.BadZipFile:
                        st.error(f"ZIP inválido: {nome}")
                else:
                    qtd, ratio, infores = indexar_arquivo_bytes(
                        area_up, nome, b, S, prototipos, embedder,
                        actor=st.session_state.usuario["username"], forcar_indexacao=forcar_up
                    )
                    if not infores["ok"]:
                        st.warning(f"{nome}: {infores['msg']}")
                    else:
                        total += qtd
                        st.success(f"{infores['msg']} | compat.: {ratio*100:.1f}% | chunks: {qtd}")
            progress.progress(1.0, text="Concluído!")
            if total > 0:
                st.success(f"✅ Total de chunks adicionados: {total}")
                
            import time
            time.sleep(5)
            st.session_state.file_uploader_key += 1
            st.rerun()

    # --- Conhecimento ---
    elif st.session_state.view == "Conhecimento":
        st.header("📚 Conhecimento por área")
        area_k = st.selectbox("Área", options=nomes_areas_ativas, key="k_area")
        idx = _carregar_indice_conhecimento()
        lista = idx.get("areas", {}).get(area_k, [])
        if not lista:
            st.info("Nenhum arquivo registrado para esta área.")
        else:
            for idx_item, item in enumerate(sorted(lista, key=lambda x: x.get("uploaded_at", ""), reverse=True)):
                with st.expander(f"📄 {item.get('filename')} — {item.get('size_bytes', 0)//1024}KB"):
                    st.caption(f"Enviado em: {item.get('uploaded_at')} | Caminho: `{item.get('saved_path','')}`")
                    motivo = st.text_input(f"Motivo para excluir", key=f"mot_{area_k}_{item.get('filename')}_{idx_item}")
                    if st.button(f"Excluir do conhecimento", key=f"del_{area_k}_{item.get('filename')}_{idx_item}"):
                        if not motivo or not motivo.strip():
                            st.error("Informe um motivo para a exclusão.")
                        else:
                            ok = deletar_documento_da_area(
                                area_k, item.get("filename"), motivo.strip(),
                                actor=st.session_state.usuario["username"]
                            )
                            if ok:
                                st.success("Excluído do conhecimento e do disco.")
                            else:
                                st.warning("Não foi possível excluir.")

    # --- Admin ---
    elif st.session_state.view == "Admin":
        st.header("🔐 Administração")
        st.subheader("Gerenciar áreas")
        cfg = carregar_config_completa_areas()
        cols_h = st.columns([3, 1, 1])
        cols_h[0].markdown("**Nome**")
        cols_h[1].markdown("**Ativo**")
        cols_h[2].markdown("**Sementes**")
        for a in cfg["areas"]:
            c = st.columns([3, 1, 1])
            c[0].write(a["nome"])
            c[1].write("✅" if a.get("ativo") else "❌")
            c[2].write(len(a.get("sementes", [])))
        st.markdown("---")
        st.markdown("**Adicionar nova área**")
        nome_novo = st.text_input("Nome da área")
        sementes_txt = st.text_area("Sementes (uma por linha)", height=80)
        if st.button("Adicionar área"):
            if not nome_novo.strip():
                st.error("Informe um nome para a área.")
            else:
                nv = {"nome": nome_novo.strip(), "ativo": True,
                      "sementes": [s.strip() for s in sementes_txt.splitlines() if s.strip()]}
                cfg["areas"].append(nv)
                salvar_config_areas(cfg, actor=st.session_state.usuario["username"],
                                    change_desc=f"ADD_AREA:{nome_novo.strip()}")
                st.success(f"Área '{nome_novo}' adicionada. Atualize a página.")

        st.markdown("---")
        area_del = st.selectbox("Excluir área", options=[a["nome"] for a in cfg["areas"]], key="sel_del_area")
        if st.button("Excluir área selecionada"):
            remov = deletar_area(area_del, actor=st.session_state.usuario["username"])
            if remov > 0:
                st.success(f"Área '{area_del}' excluída.")
            else:
                st.warning("Nada excluído.")

        st.markdown("---")
        st.markdown("**Ativar/Desativar / Editar sementes**")
        area_sel_cfg = st.selectbox("Escolha a área", options=[a["nome"] for a in cfg["areas"]])
        if area_sel_cfg:
            aobj = next((a for a in cfg["areas"] if a["nome"] == area_sel_cfg), None)
            if aobj:
                novo_estado = st.checkbox("Área ativa?", value=aobj.get("ativo", True))
                sementes_edit = st.text_area("Editar sementes", value="\n".join(aobj.get("sementes", [])), height=100)
                if st.button("Salvar alterações da área"):
                    aobj["ativo"] = novo_estado
                    aobj["sementes"] = [s.strip() for s in sementes_edit.splitlines() if s.strip()]
                    salvar_config_areas(cfg, actor=st.session_state.usuario["username"],
                                        change_desc=f"UPDATE_AREA:{area_sel_cfg}")
                    st.success("Alterações salvas.")

        st.markdown("---")
        st.subheader("Usuários (Admins)")
        with st.expander("Criar novo Admin"):
            nu = st.text_input("Novo usuário (admin)")
            np1 = st.text_input("Senha (mín. 8 caracteres)", type="password")
            np2 = st.text_input("Confirmar senha", type="password")
            if st.button("Criar Admin"):
                if not nu or not np1:
                    st.error("Usuário e senha são obrigatórios.")
                elif np1 != np2:
                    st.error("As senhas não coincidem.")
                else:
                    try:
                        criar_admin(nu.strip(), np1, actor=st.session_state.usuario["username"])
                        st.success(f"Admin '{nu}' criado!")
                    except Exception as e:
                        st.error(str(e))

        with st.expander("Alterar minha senha"):
            curr = st.text_input("Senha atual", type="password")
            new1 = st.text_input("Nova senha", type="password")
            new2 = st.text_input("Confirmar nova senha", type="password")
            if st.button("Salvar nova senha"):
                if not new1:
                    st.error("Informe a nova senha.")
                elif new1 != new2:
                    st.error("As novas senhas não coincidem.")
                else:
                    try:
                        atualizar_senha(st.session_state.usuario["username"], curr, new1,
                                        actor=st.session_state.usuario["username"])
                        st.success("Senha atualizada com sucesso.")
                    except Exception as e:
                        st.error(str(e))

    # --- Settings (editável via UI) ---
    elif st.session_state.view == "Settings":
        st.header("⚙️ Settings")
        st.caption("Edite os parâmetros do sistema diretamente pela interface.")
        s_edit = _safe_load_json(ARQUIVO_SETTINGS, DEFAULT_SETTINGS)

        col1, col2 = st.columns(2)
        with col1:
            s_edit["TEMPERATURE"] = st.slider("Temperatura LLM", 0.0, 1.0, float(s_edit.get("TEMPERATURE", 0.2)), 0.05)
            s_edit["TOP_K_CHAT"] = st.number_input("Top-K (chunks recuperados)", 1, 10, int(s_edit.get("TOP_K_CHAT", 3)))
            s_edit["MAX_TOKENS"] = st.number_input("Max tokens LLM", 400, 4000, int(s_edit.get("MAX_TOKENS", 1500)), 100)
            s_edit["MAX_HISTORY_TURNS"] = st.number_input("Histórico (pares)", 0, 10, int(s_edit.get("MAX_HISTORY_TURNS", 4)))
        with col2:
            s_edit["CHUNK_SIZE"] = st.number_input("Tamanho do chunk", 200, 4000, int(s_edit.get("CHUNK_SIZE", 1000)), 100)
            s_edit["OVERLAP"] = st.number_input("Overlap (sobreposição)", 0, 500, int(s_edit.get("OVERLAP", 100)), 10)
            s_edit["LIMITE_CTX"] = st.number_input("Limite contexto (chars)", 1000, 20000, int(s_edit.get("LIMITE_CTX", 5000)), 500)
            s_edit["FAST_MODE"] = st.checkbox("Fast Mode (usuário)", value=bool(s_edit.get("FAST_MODE", True)))

        s_edit["ENABLE_GLOBAL_SEARCH"] = st.checkbox("Busca Global", value=bool(s_edit.get("ENABLE_GLOBAL_SEARCH", False)))
        s_edit["SHOW_ADVANCED_API"] = st.checkbox("Mostrar campo de API Key", value=bool(s_edit.get("SHOW_ADVANCED_API", False)))

        if st.button("💾 Salvar settings"):
            _safe_save_json(ARQUIVO_SETTINGS, s_edit)
            log_event("UPDATE_SETTINGS", st.session_state.usuario["username"], s_edit)
            st.success("Settings salvos! Recarregue a página para aplicar.")

    # --- Logs ---
    else:
        st.header("🧾 Auditoria (somente leitura)")
        st.caption("Registros append-only com data/hora (UTC), sessão, autor, ação e detalhes.")
        filtro_acao = st.multiselect(
            "Filtrar por ação",
            ["UPLOAD_DOC", "DELETE_DOC", "CREATE_AREA", "DELETE_AREA", "UPDATE_AREAS",
             "CREATE_ADMIN", "CHANGE_PASSWORD", "TOGGLE_SETTING", "UPDATE_SETTINGS", "UPDATE_API_KEY"],
            []
        )
        filtro_area = st.text_input("Filtrar por área")
        filtro_usuario = st.text_input("Filtrar por usuário")

        registros = []
        try:
            with open(ARQUIVO_AUDIT, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        registros.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            st.error(f"Falha ao ler audit log: {e}")

        def _ok(rec):
            if filtro_acao and rec.get("action") not in filtro_acao:
                return False
            if filtro_usuario and filtro_usuario.lower() not in str(rec.get("actor", "")).lower():
                return False
            if filtro_area:
                d = rec.get("details", {})
                if filtro_area.lower() not in json.dumps(d, ensure_ascii=False).lower():
                    return False
            return True

        filtrados = sorted([r for r in registros if _ok(r)],
                           key=lambda r: r.get("ts", ""), reverse=True)
        st.info(f"Total de registros: {len(registros)} | Filtrados: {len(filtrados)}")

        if not filtrados:
            st.info("Sem registros para os filtros selecionados.")
        else:
            if st.button("⬇️ Exportar como JSON"):
                st.download_button(
                    "Download audit_log_export.json",
                    data=json.dumps(filtrados, ensure_ascii=False, indent=2),
                    file_name="audit_log_export.json",
                    mime="application/json"
                )
            for r in filtrados[:500]:
                with st.expander(f"{r.get('ts')} — {r.get('action')} — {r.get('actor')}"):
                    st.json(r)
